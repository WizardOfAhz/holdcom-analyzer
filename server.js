import express from "express";
import fetch from "node-fetch";
import cors from "cors";
import * as cheerio from "cheerio";

const app = express();
app.use(cors());
app.use(express.json({ limit: "2mb" }));

const PORT = process.env.PORT || 3000;
const GEMINI_KEY = process.env.GEMINI_KEY;

// ═══════════════════════════════════════════════════
//  GEMINI — auto-detect working model
// ═══════════════════════════════════════════════════
const MODELS = [
  "gemini-2.5-flash",
  "gemini-2.0-flash",
  "gemini-2.5-flash-lite",
  "gemini-2.0-flash-lite",
];
let _okModel = null;

async function callGemini(systemText, userText, opts = {}) {
  if (!GEMINI_KEY) throw new Error("GEMINI_KEY env var not set");

  const maxTokens = opts.maxOutputTokens || 2000;
  const temperature = opts.temperature ?? 0.3;

  const payload = {
    system_instruction: { parts: [{ text: systemText }] },
    contents: [{ role: "user", parts: [{ text: userText }] }],
    generationConfig: { maxOutputTokens: maxTokens, temperature },
  };

  // Try cached working model first, then others
  const models = _okModel
    ? [_okModel, ...MODELS.filter((m) => m !== _okModel)]
    : [...MODELS];

  const errors = [];
  for (const model of models) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${GEMINI_KEY}`;
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (res.status === 404 || res.status === 400) {
        const e = await res.json().catch(() => ({}));
        errors.push(
          `${model}: ${res.status} ${(e.error?.message || "").slice(0, 100)}`
        );
        continue;
      }

      if (res.status === 429) {
        const e = await res.json().catch(() => ({}));
        const msg = e.error?.message || "";
        // If rate limit is 0, billing not enabled — skip model
        if (msg.includes("limit") && msg.includes("0")) {
          errors.push(`${model}: rate limit 0 — billing may not be enabled`);
          continue;
        }
        // Otherwise wait and retry once
        const wait = msg.match(/retry in ([\d.]+)s/i);
        const secs = wait ? Math.ceil(parseFloat(wait[1])) + 2 : 30;
        console.log(`Rate limited on ${model}, waiting ${secs}s...`);
        await new Promise((r) => setTimeout(r, secs * 1000));
        const res2 = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (res2.ok) {
          const d2 = await res2.json();
          if (!d2.error) {
            _okModel = model;
            return extractGeminiText(d2);
          }
        }
        errors.push(`${model}: still rate limited after retry`);
        continue;
      }

      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        errors.push(
          `${model}: ${res.status} ${(e.error?.message || "").slice(0, 100)}`
        );
        continue;
      }

      const data = await res.json();
      if (data.error) {
        errors.push(`${model}: ${data.error.message.slice(0, 100)}`);
        continue;
      }

      _okModel = model;
      console.log(`Using model: ${model}`);
      return extractGeminiText(data);
    } catch (e) {
      errors.push(`${model}: ${e.message.slice(0, 100)}`);
      continue;
    }
  }

  throw new Error(
    `No working Gemini model. Tried ${models.length}:\n${errors.join("\n")}`
  );
}

function extractGeminiText(data) {
  return (data.candidates?.[0]?.content?.parts || [])
    .map((p) => p.text || "")
    .join("");
}

// ═══════════════════════════════════════════════════
//  WEBSITE FETCH — server-side, no CORS issues
// ═══════════════════════════════════════════════════
function cleanHtml(html) {
  const $ = cheerio.load(html);
  $("script, style, nav, footer, noscript, iframe, svg").remove();
  return $("body")
    .text()
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&nbsp;/g, " ")
    .replace(/\s{2,}/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim()
    .slice(0, 8000);
}

// ═══════════════════════════════════════════════════
//  ROUTES
// ═══════════════════════════════════════════════════

// Health check
app.get("/", (req, res) => {
  res.json({
    status: "ok",
    model: _okModel || "not yet detected",
    hasKey: !!GEMINI_KEY,
  });
});

// Test the API key
app.get("/test-key", async (req, res) => {
  if (!GEMINI_KEY)
    return res
      .status(500)
      .json({ ok: false, msg: "GEMINI_KEY env var not set on server" });
  try {
    const r = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models?key=${GEMINI_KEY}`
    );
    if (r.status === 401 || r.status === 403)
      return res.json({ ok: false, msg: "API key invalid or not authorized" });
    const data = await r.json();
    const names = (data.models || [])
      .map((m) => m.name.replace("models/", ""))
      .filter((n) => n.includes("gemini"));
    res.json({
      ok: true,
      msg: `Key works. Found ${names.length} Gemini models.`,
      models: names.slice(0, 8),
    });
  } catch (e) {
    res.status(500).json({ ok: false, msg: e.message });
  }
});

// Step 1: Fetch and extract website content
// Tries: direct fetch → allorigins proxy → corsproxy → Google cache
app.post("/fetch-site", async (req, res) => {
  const { url } = req.body;
  if (!url) return res.status(400).json({ error: "url required" });

  const base = url.replace(/^https?:\/\//, "").replace(/\/+$/, "");
  const fullUrl = "https://" + base;

  const browserHeaders = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "identity",
  };

  async function tryDirect(fetchUrl, label) {
    try {
      console.log(`[${label}] ${fetchUrl}`);
      const r = await fetch(fetchUrl, { headers: browserHeaders, redirect: "follow", signal: AbortSignal.timeout(20000) });
      if (!r.ok) { console.log(`[${label}] ${r.status}`); return null; }
      const html = await r.text();
      const text = cleanHtml(html);
      if (text.length < 100) { console.log(`[${label}] too short: ${text.length}`); return null; }
      console.log(`[${label}] OK: ${text.length} chars`);
      return text;
    } catch (e) { console.log(`[${label}] ${e.message}`); return null; }
  }

  // Strategy 1: Direct
  let text = await tryDirect(fullUrl, "direct");
  if (!text) text = await tryDirect("https://www." + base.replace(/^www\./, ""), "direct-www");

  // Strategy 2: allorigins proxy
  if (!text) {
    try {
      console.log("[allorigins] trying...");
      const r = await fetch("https://api.allorigins.win/get?url=" + encodeURIComponent(fullUrl), { signal: AbortSignal.timeout(20000) });
      if (r.ok) {
        const json = await r.json();
        if (json.contents && json.contents.length > 200) {
          const t = cleanHtml(json.contents);
          if (t.length >= 100) { text = t; console.log(`[allorigins] OK: ${t.length} chars`); }
        }
      }
    } catch (e) { console.log("[allorigins]", e.message); }
  }

  // Strategy 3: corsproxy
  if (!text) {
    try {
      console.log("[corsproxy] trying...");
      const r = await fetch("https://corsproxy.io/?" + encodeURIComponent(fullUrl), { headers: browserHeaders, signal: AbortSignal.timeout(20000) });
      if (r.ok) {
        const html = await r.text();
        if (html.length > 200) {
          const t = cleanHtml(html);
          if (t.length >= 100) { text = t; console.log(`[corsproxy] OK: ${t.length} chars`); }
        }
      }
    } catch (e) { console.log("[corsproxy]", e.message); }
  }

  // Strategy 4: Google cache
  if (!text) {
    try {
      console.log("[google-cache] trying...");
      const r = await fetch("https://webcache.googleusercontent.com/search?q=cache:" + encodeURIComponent(fullUrl), { headers: browserHeaders, redirect: "follow", signal: AbortSignal.timeout(20000) });
      if (r.ok) {
        const html = await r.text();
        const t = cleanHtml(html);
        if (t.length >= 100) { text = t; console.log(`[google-cache] OK: ${t.length} chars`); }
      }
    } catch (e) { console.log("[google-cache]", e.message); }
  }

  if (text) {
    res.json({ text, success: true });
  } else {
    console.error("All strategies failed for:", url);
    res.json({ text: "", success: false, error: "Site blocked all fetch methods. Paste content manually." });
  }
});

// Step 2: Extract structured info from website text via Gemini
app.post("/extract", async (req, res) => {
  const { siteText, clientUrl } = req.body;
  try {
    const result = await callGemini(
      "You are a business analyst. Extract key information from the provided website text. Report only what is in the text. Never invent or assume.",
      `Website: ${clientUrl}\n\nPage content:\n${siteText}\n\nExtract:\n1. BUSINESS NAME & TYPE\n2. LOCATION (city, state)\n3. SERVICES (list every one)\n4. EVENT TYPES\n5. KEY PHRASES & TAGLINES (direct quotes in "")\n6. VALUE PROPOSITIONS\n7. CALLS TO ACTION\n8. NOTABLE DETAILS (years, awards, venues, news)`,
      { maxOutputTokens: 1500, temperature: 0.1 }
    );
    res.json({ result });
  } catch (e) {
    console.error("extract error:", e.message);
    res.status(500).json({ error: e.message });
  }
});

// Step 3: Gap analysis — returns JSON scorecard + insights + suggestions
app.post("/analyze", async (req, res) => {
  const { clientName, clientUrl, siteText, script } = req.body;
  try {
    const prompt =
      `CLIENT: ${clientName}\nURL: ${clientUrl}` +
      `\n\nWEBSITE CONTENT:\n${siteText}` +
      `\n\nCURRENT ON-HOLD SCRIPT:\n${script}` +
      `\n\nReturn ONLY this JSON (values under 15 words, arrays max 5 items):\n` +
      `{"scorecard":{"audience_alignment":{"score":0,"basis":""},"offer_coverage":{"score":0,"basis":""},"conversion_readiness":{"score":0,"basis":""},"content_freshness":{"score":0,"basis":""}},"insights":[{"observation":"","evidence":"","implication":""}],"suggestions":[{"action":"","basis":"","priority":"high"}]}` +
      `\n\nScoring: audience_alignment=% of website audiences in script, offer_coverage=% of services in script, conversion_readiness=50 if CTA exists +10 per action step, content_freshness=100 minus 20 per dated element.` +
      `\nGive exactly 3 insights with direct evidence. Give exactly 4 suggestions.`;

    const raw = await callGemini(
      "You are a messaging analyst. Return ONLY valid JSON starting with {. No preamble.",
      prompt,
      { maxOutputTokens: 3000, temperature: 0.2 }
    );

    // Parse JSON from response (Gemini sometimes adds preamble)
    const s = raw.indexOf("{"),
      e = raw.lastIndexOf("}");
    if (s === -1 || e === -1)
      throw new Error("No JSON found in Gemini response");
    const data = JSON.parse(raw.slice(s, e + 1));
    res.json({ result: data, raw });
  } catch (e) {
    console.error("analyze error:", e.message);
    res.status(500).json({ error: e.message });
  }
});

// Step 4: Script rewrite
app.post("/rewrite", async (req, res) => {
  const { clientName, wordLimit, script, gaps, siteText } = req.body;
  try {
    const prompt =
      `CLIENT: ${clientName}\nWORD LIMIT: ${wordLimit} words (hard limit — count carefully)\n\n` +
      `ORIGINAL SCRIPT:\n${script}\n\n` +
      `GAPS TO WEAVE IN:\n${gaps.map((g, i) => i + 1 + ". " + g).join("\n")}\n\n` +
      `WEBSITE CONTEXT:\n${(siteText || "").slice(0, 600)}\n\n` +
      `Write the complete revised script. Wrap new/changed phrases in <<NEW>>phrase<</NEW>>. ` +
      `Then write ===CHANGES=== followed by bullet points of what changed.`;

    const sys =
      `You are a professional on-hold messaging copywriter at Holdcom.\n` +
      `Write a COMPLETE revised script — not the original with things bolted on.\n` +
      `Weave new elements in naturally so the result reads as one cohesive piece.\n` +
      `Match original tone and voice exactly. Hard word limit: ${wordLimit} words.\n` +
      `Wrap only NEW or significantly changed phrases in <<NEW>>...</</NEW>>.\n` +
      `A senior copywriter should not be able to find the seams.`;

    const raw = await callGemini(sys, prompt, {
      maxOutputTokens: 3500,
      temperature: 0.4,
    });
    res.json({ result: raw });
  } catch (e) {
    console.error("rewrite error:", e.message);
    res.status(500).json({ error: e.message });
  }
});

// Step 4b: Script revision
app.post("/revise", async (req, res) => {
  const { draft, input, wordLimit } = req.body;
  try {
    const raw = await callGemini(
      `You are a Holdcom on-hold messaging copywriter. Revise the script per the feedback. Same rules: full cohesive script, ${wordLimit} word limit, wrap changes in <<NEW>><</NEW>>, end with ===CHANGES===.`,
      `CURRENT SCRIPT:\n${draft}\n\nREVISION:\n${input}`,
      { maxOutputTokens: 2500, temperature: 0.4 }
    );
    res.json({ result: raw });
  } catch (e) {
    console.error("revise error:", e.message);
    res.status(500).json({ error: e.message });
  }
});

// Step 5: Email generation
app.post("/email", async (req, res) => {
  const { clientName, repName, finding, clientId } = req.body;
  try {
    const sys =
      `You are a Holdcom account manager writing a short client outreach email.\n` +
      `Rules: warm, direct, not salesy. Never mention AI tools by name — say "we analyzed your website".\n` +
      `Reference 1-2 specific things from their site. Mention a scorecard and draft are ready.\n` +
      `Soft CTA. Under 130 words in body. Sign off with rep name.\n` +
      `Format: Subject: [subject] then blank line then body only.`;

    const raw = await callGemini(
      sys,
      `Client: ${clientName}\nRep: ${repName}\nKey finding: ${finding}\nLink: https://insights.holdcom.com/${clientId}`,
      { maxOutputTokens: 600, temperature: 0.5 }
    );
    res.json({ result: raw });
  } catch (e) {
    console.error("email error:", e.message);
    res.status(500).json({ error: e.message });
  }
});

// ═══════════════════════════════════════════════════
app.listen(PORT, () => {
  console.log(`Holdcom Analyzer API running on port ${PORT}`);
  console.log(`Gemini key: ${GEMINI_KEY ? "✓ set" : "✗ MISSING — set GEMINI_KEY env var"}`);
});
