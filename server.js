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
  "gemini-2.5-flash-lite",
  "gemini-2.5-pro",
  "gemini-2.5-flash-preview-05-20",
];
let _okModel = null;

async function callGemini(systemText, userText, opts = {}) {
  if (!GEMINI_KEY) throw new Error("GEMINI_KEY env var not set");

  const maxTokens = opts.maxOutputTokens || 2000;
  const temperature = opts.temperature ?? 0.3;
  const timeoutMs = opts.timeoutMs || 60000; // 60s default

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

    // Try up to 3 times per model (handles 503 temporary overload)
    for (let attempt = 1; attempt <= 3; attempt++) {
      try {
        if (attempt > 1) console.log(`${model}: retry attempt ${attempt}...`);

        // Add abort controller for timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
        const startTime = Date.now();

        const res = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });
        clearTimeout(timeoutId);
        console.log(`${model}: response in ${Date.now() - startTime}ms (status ${res.status})`);

        if (res.status === 404 || res.status === 400) {
          const e = await res.json().catch(() => ({}));
          errors.push(
            `${model}: ${res.status} ${(e.error?.message || "").slice(0, 100)}`
          );
          break; // Don't retry 404/400 — model is gone
        }

        if (res.status === 503) {
          console.log(`${model}: 503 overloaded (attempt ${attempt}/3)`);
          if (attempt < 3) {
            await new Promise((r) => setTimeout(r, 5000 * attempt)); // 5s, 10s
            continue; // Retry same model
          }
          errors.push(`${model}: 503 overloaded after 3 attempts`);
          break; // Move to next model
        }

        if (res.status === 429) {
          const e = await res.json().catch(() => ({}));
          const msg = e.error?.message || "";
          if (msg.includes("limit") && msg.includes("0")) {
            errors.push(`${model}: rate limit 0 — billing may not be enabled`);
            break;
          }
          const wait = msg.match(/retry in ([\d.]+)s/i);
          const secs = wait ? Math.ceil(parseFloat(wait[1])) + 2 : 30;
          console.log(`Rate limited on ${model}, waiting ${secs}s...`);
          await new Promise((r) => setTimeout(r, secs * 1000));
          continue; // Retry same model
        }

        if (!res.ok) {
          const e = await res.json().catch(() => ({}));
          errors.push(
            `${model}: ${res.status} ${(e.error?.message || "").slice(0, 100)}`
          );
          break;
        }

        const data = await res.json();
        if (data.error) {
          errors.push(`${model}: ${data.error.message.slice(0, 100)}`);
          break;
        }

        const text = extractGeminiText(data);
        if (!text || text.length < 10) {
          console.log(`${model}: returned empty/short response`, JSON.stringify(data).slice(0, 300));
          errors.push(`${model}: empty response`);
          break;
        }

        _okModel = model;
        console.log(`Using model: ${model}, returned ${text.length} chars`);
        return text;

      } catch (e) {
        if (e.name === 'AbortError') {
          console.log(`${model}: TIMEOUT after ${timeoutMs}ms`);
          errors.push(`${model}: timeout (${timeoutMs/1000}s)`);
        } else {
          errors.push(`${model}: ${e.message.slice(0, 100)}`);
        }
        break;
      }
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
    .replace(/©.*?\d{4}/g, "")
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
// Fetches homepage + common subpages, combines all text
app.post("/fetch-site", async (req, res) => {
  const { url } = req.body;
  if (!url) return res.status(400).json({ error: "url required" });

  const base = url.replace(/^https?:\/\//, "").replace(/\/+$/, "");
  const origin = "https://" + base;

  const browserHeaders = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "identity",
  };

  // Fetch a single page — try direct then corsproxy
  async function fetchPage(pageUrl) {
    // Try direct first
    try {
      const r = await fetch(pageUrl, { headers: browserHeaders, redirect: "follow", signal: AbortSignal.timeout(15000) });
      if (r.ok) {
        const html = await r.text();
        const t = cleanHtml(html);
        if (t.length >= 50) return t;
      }
    } catch (e) { /* fall through */ }

    // Try corsproxy
    try {
      const r = await fetch("https://corsproxy.io/?" + encodeURIComponent(pageUrl), { headers: browserHeaders, signal: AbortSignal.timeout(15000) });
      if (r.ok) {
        const html = await r.text();
        const t = cleanHtml(html);
        if (t.length >= 50) return t;
      }
    } catch (e) { /* fall through */ }

    return null;
  }

  // Minimal fallback paths — the link discovery above finds the real pages
  // These are just universal paths that almost every business site has
  const subpages = [
    "/about", "/about-us", "/services", "/contact"
  ];

  console.log(`[multi-page] Starting crawl of ${origin}`);

  // Fetch homepage first to confirm the site works
  const homeText = await fetchPage(origin + "/");
  if (!homeText) {
    console.error("[multi-page] Could not fetch homepage for:", url);
    return res.json({ text: "", success: false, error: "Could not fetch site. Paste content manually." });
  }

  // Now try to discover actual links from the homepage HTML
  let discoveredPaths = [];
  try {
    // Fetch raw HTML to find links
    let rawHtml = "";
    try {
      const r = await fetch("https://corsproxy.io/?" + encodeURIComponent(origin + "/"), { headers: browserHeaders, signal: AbortSignal.timeout(15000) });
      if (r.ok) rawHtml = await r.text();
    } catch (e) { /* skip discovery */ }

    if (rawHtml) {
      const $ = cheerio.load(rawHtml);
      $("a[href]").each((_, el) => {
        const href = $(el).attr("href") || "";
        // Only internal links, not anchors or external
        if (href.startsWith("/") && !href.startsWith("//") && href.length > 1 && href.length < 60) {
          const path = href.split("?")[0].split("#")[0];
          if (!discoveredPaths.includes(path)) discoveredPaths.push(path);
        }
      });
      console.log(`[multi-page] Discovered ${discoveredPaths.length} internal links`);
    }
  } catch (e) { /* skip discovery */ }

  // Merge discovered paths with common subpages, deduplicate
  const allPaths = [...new Set([...discoveredPaths, ...subpages])].filter(p => p !== "/");

  // Fetch subpages in parallel (max 6 at a time to be polite)
  const pageTexts = { "HOME": homeText };
  const batchSize = 6;

  for (let i = 0; i < allPaths.length && Object.keys(pageTexts).length < 10; i += batchSize) {
    const batch = allPaths.slice(i, i + batchSize);
    const results = await Promise.allSettled(
      batch.map(async (path) => {
        const t = await fetchPage(origin + path);
        return { path, text: t };
      })
    );
    for (const r of results) {
      if (r.status === "fulfilled" && r.value.text && r.value.text.length >= 100) {
        // Skip if it's basically the same as the homepage (some sites return same content for all paths)
        if (r.value.text.slice(0, 200) !== homeText.slice(0, 200)) {
          const label = r.value.path.replace(/\//g, "").toUpperCase() || "PAGE";
          pageTexts[label] = r.value.text;
          console.log(`[multi-page] ${r.value.path} — ${r.value.text.length} chars`);
        }
      }
    }
  }

  // Combine all page texts with labels, cap total at 12000 chars for Gemini
  let combined = "";
  for (const [label, text] of Object.entries(pageTexts)) {
    combined += `\n\n=== ${label} PAGE ===\n${text}`;
    if (combined.length > 12000) break;
  }
  combined = combined.trim().slice(0, 12000);

  console.log(`[multi-page] Done. ${Object.keys(pageTexts).length} pages, ${combined.length} total chars`);
  res.json({ text: combined, success: true });
});

// Step 2: Extract structured info from website text via Gemini
app.post("/extract", async (req, res) => {
  const { siteText, clientUrl } = req.body;
  try {
    const result = await callGemini(
      "You are a business analyst. Extract key information from the provided website text. CRITICAL: Report ONLY what is explicitly stated in the text below. NEVER invent, assume, or add details not present. If something is not mentioned, say 'Not mentioned'. Do NOT guess the location, do NOT add descriptors like 'waterfront' or 'oceanfront' unless those exact words appear in the text.",
      `Website: ${clientUrl}\n\nPage content:\n${siteText}\n\nExtract:\n1. BUSINESS NAME & TYPE\n2. LOCATION (city, state — ONLY if explicitly stated)\n3. SERVICES (list every one mentioned)\n4. EVENT TYPES (only those explicitly listed)\n5. KEY PHRASES & TAGLINES (direct quotes in "" — must appear verbatim in text)\n6. VALUE PROPOSITIONS\n7. CALLS TO ACTION\n8. NOTABLE DETAILS (years, awards, venues, news — only if stated)`,
      { maxOutputTokens: 1500, temperature: 0.0 }
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
  const { clientName, industry, wordLimit, script, gaps, siteText, repNotes } = req.body;
  console.log(`[rewrite] ${clientName} | industry: ${industry} | ${gaps?.length || 0} gaps | repNotes: ${repNotes ? 'yes' : 'no'}`);
  try {
    const prompt =
      `CLIENT: ${clientName}\nINDUSTRY: ${industry || 'General'}\nWORD LIMIT: ${wordLimit} words (hard limit — count carefully)\n\n` +
      `ORIGINAL SCRIPT (for tone/voice reference only):\n${script}\n\n` +
      `GAPS TO WEAVE IN:\n${gaps.map((g, i) => i + 1 + ". " + g).join("\n")}\n\n` +
      `WEBSITE CONTEXT:\n${(siteText || "").slice(0, 800)}\n\n` +
      (repNotes ? `REP NOTES (additional context from the account manager — treat as high priority):\n${repNotes}\n\n` : '') +
      `Write the complete script from scratch. Wrap new/changed phrases in <<NEW>>phrase<</NEW>>. ` +
      `Then write ===CHANGES=== followed by bullet points of what changed.`;

    const sys =
      `You are a senior on-hold messaging copywriter at Holdcom, writing for a ${industry || 'business'} client.\n` +
      `APPROACH: Write a FRESH script from scratch — do NOT edit or patch the original. Use the original script only to understand the client's brand voice and tone. Then write something new that sounds like the same brand but is a complete reimagining.\n` +
      `The script should feel like it was written by a professional who studied the client's website deeply — not like a revision of the old script.\n` +
      `Hard word limit: ${wordLimit} words. Count carefully.\n` +
      `FORMAT: Use standard on-hold script format:\n` +
      `- Number each paragraph with bracketed numbers: [1], [2], [3], etc.\n` +
      `- Insert {music} between paragraphs to indicate music breaks\n` +
      `- Match the paragraph/music structure of the original script if one was provided\n` +
      `Wrap only NEW or significantly changed phrases in <<NEW>>...</<NEW>>. CRITICAL: use ONLY <<NEW>> to open and <</NEW>> to close. Do NOT use <<END NEW>>, <<NEW END>>, or any other closing variant.\n` +
      `CRITICAL: ONLY use facts, services, and descriptions from the website context and rep notes provided. NEVER invent features, locations, or descriptors (like "waterfront", "oceanfront", "lakeside") that are not explicitly in the source material.\n` +
      `If rep notes mention social media activity, events, or services not on the website, you may include those — they come from the account manager who knows the client.`;

    const raw = await callGemini(sys, prompt, {
      maxOutputTokens: 3500,
      temperature: 0.5,
      timeoutMs: 90000, // 90s — rewrite can take a while
    });
    console.log(`[rewrite] SUCCESS: ${raw.length} chars`);
    res.json({ result: raw });
  } catch (e) {
    console.error("[rewrite] ERROR:", e.message);
    res.status(500).json({ error: e.message });
  }
});

// Step 4b: Script revision
app.post("/revise", async (req, res) => {
  const { draft, input, wordLimit } = req.body;
  try {
    const raw = await callGemini(
      `You are a Holdcom on-hold messaging copywriter. Revise the script per the feedback. Same rules: full cohesive script, ${wordLimit} word limit, use [1] [2] [3] paragraph numbering, wrap only changed phrases in <<NEW>>...</<NEW>> (no other bracket styles), end with ===CHANGES=== followed by bullet points of what changed.`,
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
  const { clientName, repName, finding, scores, clientId, link } = req.body;
  try {
    const sys =
      `You are a Holdcom account manager writing a short, curiosity-driven client outreach email.\n` +
      `GOAL: Get them to click the link to see their personalized scorecard and draft script.\n\n` +
      `RULES:\n` +
      `- Warm, direct, human — NOT salesy, NOT robotic, NOT generic\n` +
      `- Never mention AI, ChatGPT, Gemini, or any AI tools — say "we analyzed your website" or "our team reviewed"\n` +
      `- Reference 1-2 SPECIFIC things from their site (a service, a phrase, something real)\n` +
      `- Tease the scorecard: mention they scored well in some areas but there are gaps — don't give all the numbers, create curiosity\n` +
      `- Include the landing page link naturally — "I put together a quick scorecard for you: [link]"\n` +
      `- Mention a draft script update is ready to review together\n` +
      `- Soft CTA: suggest a 10-minute call to walk through it, no pressure\n` +
      `- Under 130 words in body\n` +
      `- Sign off with rep name only (no title, no company)\n` +
      `- Format: Subject: [subject line] then blank line then body only\n` +
      `- ONLY reference facts provided below. NEVER invent locations, features, or details.`;

    const raw = await callGemini(
      sys,
      `Client: ${clientName}\nRep: ${repName}\nScorecard results: ${scores}\nKey findings: ${finding}\nLanding page link: ${link}`,
      { maxOutputTokens: 600, temperature: 0.5 }
    );
    res.json({ result: raw });
  } catch (e) {
    console.error("email error:", e.message);
    res.status(500).json({ error: e.message });
  }
});

// ═══════════════════════════════════════════════════
//  TRACK LANDING PAGE EVENTS (Pipedream-ready)
// ═══════════════════════════════════════════════════
// To activate: set PIPEDREAM_URL env var on Render to your Pipedream webhook URL.
// Pipedream will receive: event, client, ts (and optionally type for feedback).
// From there, configure Pipedream to send email or Teams/Slack notification to rep.
const PIPEDREAM_URL = process.env.PIPEDREAM_URL || '';

app.post('/track', async (req, res) => {
  const { event, client, type, ts } = req.body;
  console.log(`[track] event=${event} client=${client || '?'} type=${type || '-'} ts=${ts || new Date().toISOString()}`);

  if (PIPEDREAM_URL) {
    try {
      const params = new URLSearchParams({ event, client: client || '', ts: ts || new Date().toISOString() });
      if (type) params.set('type', type);
      await fetch(`${PIPEDREAM_URL}?${params.toString()}`);
    } catch (e) {
      console.error('[track] Pipedream forward failed:', e.message);
    }
  }

  res.sendStatus(200);
});

// ═══════════════════════════════════════════════════
//  PUBLISH LANDING PAGE TO NETLIFY
// ═══════════════════════════════════════════════════
const NETLIFY_TOKEN = process.env.NETLIFY_TOKEN;
const NETLIFY_SITE_ID = process.env.NETLIFY_SITE_ID;

app.post("/publish", async (req, res) => {
  const { html, clientId } = req.body;
  if (!html || !clientId) return res.status(400).json({ error: "html and clientId required" });
  if (!NETLIFY_TOKEN || !NETLIFY_SITE_ID) return res.status(500).json({ error: "Netlify not configured on server" });

  try {
    // Netlify file digest deploy: we create a simple deploy with the file content
    const fileName = clientId + ".html";
    const fileContent = html;
    const encoder = new TextEncoder();
    const fileBytes = encoder.encode(fileContent);

    // Step 1: Calculate SHA1 hash of the file
    const hashBuffer = await crypto.subtle.digest("SHA-1", fileBytes);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const sha1 = hashArray.map(b => b.toString(16).padStart(2, "0")).join("");

    // Step 2: Create deploy with file manifest
    const deployRes = await fetch(`https://api.netlify.com/api/v1/sites/${NETLIFY_SITE_ID}/deploys`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${NETLIFY_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        files: {
          ["/" + fileName]: sha1,
          "/index.html": sha1  // Also update the index to show latest
        },
        draft: false,
      }),
    });

    if (!deployRes.ok) {
      const err = await deployRes.text();
      throw new Error("Netlify deploy failed: " + err);
    }

    const deploy = await deployRes.json();
    const deployId = deploy.id;

    // Step 3: Upload the file content
    const uploadRes = await fetch(`https://api.netlify.com/api/v1/deploys/${deployId}/files/${encodeURIComponent("/" + fileName)}`, {
      method: "PUT",
      headers: {
        "Authorization": `Bearer ${NETLIFY_TOKEN}`,
        "Content-Type": "application/octet-stream",
      },
      body: fileContent,
    });

    if (!uploadRes.ok) {
      const err = await uploadRes.text();
      throw new Error("Netlify upload failed: " + err);
    }

    // Also upload as index.html
    await fetch(`https://api.netlify.com/api/v1/deploys/${deployId}/files//index.html`, {
      method: "PUT",
      headers: {
        "Authorization": `Bearer ${NETLIFY_TOKEN}`,
        "Content-Type": "application/octet-stream",
      },
      body: fileContent,
    });

    const siteUrl = deploy.ssl_url || deploy.url || `https://holdcom-insights.netlify.app`;
    const pageUrl = siteUrl + "/" + fileName;

    console.log(`Published: ${pageUrl}`);
    res.json({ url: pageUrl, siteUrl, deployId });
  } catch (e) {
    console.error("publish error:", e.message);
    res.status(500).json({ error: e.message });
  }
});

// ═══════════════════════════════════════════════════
app.listen(PORT, () => {
  console.log(`Holdcom Analyzer API running on port ${PORT}`);
  console.log(`Gemini key: ${GEMINI_KEY ? "✓ set" : "✗ MISSING — set GEMINI_KEY env var"}`);
  console.log(`Netlify: ${NETLIFY_TOKEN ? "✓ configured" : "✗ MISSING — set NETLIFY_TOKEN and NETLIFY_SITE_ID"}`);
});
