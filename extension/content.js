/**
 * DeepShield — Content Script (Floating Overlay Edition)
 *
 * Works on ALL websites including Instagram Reels, TikTok, YouTube Shorts.
 * Uses position:fixed overlays — never clipped by overflow:hidden.
 * On social media: captures canvas frames (avoids auth-gated CDN URL issues).
 */

const BACKEND = "http://localhost:8000";
const MIN_SIZE = 80;
const MAX_CONCURRENT = 3;
const SOCIAL_HOSTS = /instagram|facebook|tiktok|youtube|twitter|x\.com/i;
const IS_SOCIAL = SOCIAL_HOSTS.test(location.hostname);

/* ─── State ─────────────────────────────────────────────────────────────── */
const seen = new WeakMap();   // el → { btn, badge }
const results = new Map();       // url → result
const videoUrls = new WeakMap();   // video el → real CDN url
let scanning = 0;

/* ─── Intercept fetch() for video CDN URLs ──────────────────────────────── */
if (SOCIAL_HOSTS.test(location.hostname)) {
    const _fetch = window.fetch.bind(window);
    window.fetch = function (input, init) {
        const url = typeof input === "string" ? input : (input?.url || "");
        if (url.startsWith("http") && /\.(mp4|webm|m4v)(\?|$)/i.test(url)) {
            setTimeout(() => {
                document.querySelectorAll("video").forEach(v => {
                    if (!videoUrls.has(v)) videoUrls.set(v, url);
                });
            }, 100);
        }
        return _fetch(input, init);
    };
}

/* ─── Extract video URL from Instagram's embedded page JSON ─────────────
   Instagram always embeds the real CDN URL in inline <script> tags.
   This is the most reliable method for Reels.
   ─────────────────────────────────────────────────────────────────────── */
function getInstagramVideoUrlFromPage() {
    const patterns = [
        /"video_url"\s*:\s*"(https:[^"]+)"/,
        /"playback_url"\s*:\s*"(https:[^"]+)"/,
        /"contentUrl"\s*:\s*"(https:[^"]+\.mp4[^"]*)"/,
        /(https:\/\/[a-z0-9\-]+\.cdninstagram\.com\/[^\s"'<>]+\.mp4[^\s"'<>]*)/,
        /(https:\/\/[a-z0-9\-]+\.fbcdn\.net\/[^\s"'<>]+\.mp4[^\s"'<>]*)/,
    ];

    for (const script of document.querySelectorAll("script")) {
        const text = script.textContent || "";
        if (!text.includes("mp4") && !text.includes("video_url")) continue;
        for (const pat of patterns) {
            const m = text.match(pat);
            if (m) {
                return m[1]
                    .replace(/\\u0026/g, "&")
                    .replace(/\\\//g, "/")
                    .replace(/\\/g, "");
            }
        }
    }

    // Also check window.__additionalData (Instagram SPA)
    try {
        const ad = JSON.stringify(window.__additionalData || {});
        for (const pat of patterns) {
            const m = ad.match(pat);
            if (m) return m[1].replace(/\\u0026/g, "&").replace(/\\\//g, "/");
        }
    } catch (_) { }

    return null;
}

/* ─── Resolve real URL for any media element ────────────────────────────── */
function getRealUrl(el) {
    if (el.tagName === "VIDEO") {
        // 1. Previously intercepted CDN URL
        if (videoUrls.has(el)) return videoUrls.get(el);

        // 2. data attributes (Instagram/TikTok inject these)
        for (const attr of ["data-video-url", "data-src", "data-original-src"]) {
            const v = el.getAttribute(attr);
            if (v && v.startsWith("http")) return v;
        }

        // 3. <source> child elements
        for (const s of el.querySelectorAll("source")) {
            const v = s.src || s.getAttribute("src") || "";
            if (v && v.startsWith("http")) return v;
        }

        // 4. Direct non-blob src
        if (el.src && !el.src.startsWith("blob:") && el.src.startsWith("http")) {
            return el.src;
        }

        // 5. Parse Instagram's embedded page JSON (most reliable for Reels)
        if (/instagram|facebook/i.test(location.hostname)) {
            const fromPage = getInstagramVideoUrlFromPage();
            if (fromPage) {
                videoUrls.set(el, fromPage);
                return fromPage;
            }
        }

        return null;
    }

    // Images / Audio
    const src = el.src || el.currentSrc || el.getAttribute("src") || "";
    return (src && src.startsWith("http")) ? src : null;
}

/* ─── Overlay creation helper ───────────────────────────────────────────── */
function createOverlayEl(cls, html) {
    const el = document.createElement("div");
    el.className = cls;
    el.innerHTML = html;
    el.dataset.ds = "overlay";
    document.body.appendChild(el);
    return el;
}

/* ─── Canvas frame capture (for social media videos) ───────────────────────── */
async function captureVideoFrame(videoEl) {
    return new Promise((resolve, reject) => {
        try {
            const w = videoEl.videoWidth || 640;
            const h = videoEl.videoHeight || 360;
            if (w === 0 || h === 0) { reject(new Error("video not ready")); return; }
            const canvas = document.createElement("canvas");
            canvas.width = Math.min(w, 1280);
            canvas.height = Math.min(h, 720);
            const ctx = canvas.getContext("2d");
            ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
            resolve(canvas.toDataURL("image/jpeg", 0.85));
        } catch (err) {
            reject(err);
        }
    });
}

/* ─── Send scan request via background service worker ───────────────────────── */
// Content scripts can't reliably fetch localhost — route through background.
function requestScan(payload) {
    return new Promise((resolve, reject) => {
        chrome.runtime.sendMessage({ type: "DO_SCAN", payload }, (resp) => {
            if (chrome.runtime.lastError) { reject(new Error(chrome.runtime.lastError.message)); return; }
            if (resp?.error) reject(new Error(resp.error));
            else resolve(resp?.data);
        });
    });
}

/* ─── Send frame to backend via background ───────────────────────────────────── */
async function scanViaFrame(el) {
    const cacheKey = "__frame__" + location.href;
    if (results.has(cacheKey)) { renderResult(el, results.get(cacheKey)); return; }

    scanning++;
    showBadge(el, "scanning", '<span class="ds-spinner"></span> Scanning frame…');
    try {
        if (el.readyState < 2) {
            await new Promise(r => el.addEventListener("loadeddata", r, { once: true }));
        }
        const frameData = await captureVideoFrame(el);
        const data = await requestScan({
            endpoint: "analyze-image-data",
            body: { image_data: frameData, source_url: location.href },
        });
        results.set(cacheKey, data);
        chrome.runtime.sendMessage({ type: "RESULT", result: data, url: location.href });
        renderResult(el, data);
    } catch (e) {
        const raw = e.message || "";
        const msg = raw.includes("fetch") || raw.includes("Could not establish") ? "Backend offline"
            : raw.includes("tainted") ? "Frame blocked (CORS)"
                : raw.includes("not ready") ? "Video not loaded yet"
                    : raw.slice(0, 40);
        showBadge(el, "error", `⚡ ${msg}`);
    } finally {
        scanning--;
    }
}

/* ─── Scanning logic ────────────────────────────────────────────────────────── */
async function scan(el) {
    const state = seen.get(el);
    if (!state) return;
    if (scanning >= MAX_CONCURRENT) { showBadge(el, "error", "⚡ Busy—retry"); return; }

    // Social media VIDEO → always use canvas frame capture (CDN URLs are auth-gated)
    if (IS_SOCIAL && el.tagName === "VIDEO") {
        return scanViaFrame(el);
    }

    const url = getRealUrl(el);
    if (!url) {
        if (el.tagName === "VIDEO") return scanViaFrame(el);
        showBadge(el, "error", "⚡ URL not found"); return;
    }
    if (results.has(url)) { renderResult(el, results.get(url)); return; }

    scanning++;
    showBadge(el, "scanning", '<span class="ds-spinner"></span> Scanning…');
    const type = el.tagName === "VIDEO" ? "video" : el.tagName === "AUDIO" ? "audio" : "image";
    try {
        const data = await requestScan({
            endpoint: `analyze-${type}`,
            body: { url },
        });
        results.set(url, data);
        chrome.runtime.sendMessage({ type: "RESULT", result: data, url });
        renderResult(el, data);
    } catch (e) {
        const msg = e.message.includes("fetch") || e.message.includes("Could not establish")
            ? "Backend offline" : e.message.slice(0, 40);
        showBadge(el, "error", `⚡ ${msg}`);
    } finally {
        scanning--;
    }
}

/* ─── Badge helpers ─────────────────────────────────────────────────────── */
function getBadgePos(el) {
    const r = el.getBoundingClientRect();
    return {
        top_badge: r.bottom - 36,
        left_badge: r.left + 6,
        top_btn: r.top + 6,
        left_btn: r.right - 80,       // ~80px wide button
    };
}

function showBadge(el, kind, html) {
    const state = seen.get(el);
    if (!state) return;
    state.badge?.remove();

    const cls = kind === "scanning"
        ? "ds-result-overlay ds-result-scanning"
        : "ds-result-overlay ds-result-medium";

    const badge = createOverlayEl(cls, html);
    state.badge = badge;

    const p = getBadgePos(el);
    badge.style.top = `${p.top_badge}px`;
    badge.style.left = `${p.left_badge}px`;
}

function renderResult(el, result) {
    const state = seen.get(el);
    if (!state) return;
    state.badge?.remove();

    const r = result.risk_level || "Low";
    const s = Math.round(result.authenticity_score ?? 0);
    const icon = { Low: "✅", Medium: "⚠️", High: "🔴" }[r] || "✅";
    const cls = { Low: "ds-result-low", Medium: "ds-result-medium", High: "ds-result-high" }[r];

    const badge = createOverlayEl(
        `ds-result-overlay ${cls}`,
        `${icon} <span>${r} · ${s}/100</span>`
    );
    state.badge = badge;

    const p = getBadgePos(el);
    badge.style.top = `${p.top_badge}px`;
    badge.style.left = `${p.left_badge}px`;

    if (r === "High") el.classList.add("ds-high-risk");
}

/* ─── Attach scan button ────────────────────────────────────────────────── */
function attach(el) {
    if (seen.has(el)) return;

    if (el.tagName === "IMG") {
        const src = el.getAttribute("src") || el.src || "";
        if (!src || src.startsWith("data:")) return;
    }

    // Only skip if we KNOW it's tiny (not just not-yet-rendered)
    const rect = el.getBoundingClientRect();
    const w = rect.width || el.offsetWidth || el.clientWidth;
    if (w > 0 && w < MIN_SIZE) return;

    seen.set(el, { btn: null, badge: null });

    const btn = createOverlayEl("ds-scan-overlay", "🛡 Scan");
    seen.get(el).btn = btn;

    const updatePos = () => {
        const r = el.getBoundingClientRect();
        if (!r.width) return;
        btn.style.top = `${r.top + 6}px`;
        btn.style.left = `${r.right - 88}px`;
        const b = seen.get(el)?.badge;
        if (b) {
            b.style.top = `${r.bottom - 36}px`;
            b.style.left = `${r.left + 6}px`;
        }
    };

    btn.addEventListener("click", (e) => {
        e.stopPropagation();
        e.preventDefault();
        scan(el);
    });

    // On social media: only show the button for the PLAYING video (not preloaded ones)
    new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            let show = entry.isIntersecting;
            // If social media VIDEO, only show button when it's the active (playing/focused) one
            if (show && IS_SOCIAL && el.tagName === "VIDEO") {
                // Hide if video is paused AND another video on the page is playing
                const anyPlaying = [...document.querySelectorAll("video")].some(
                    v => v !== el && !v.paused && !v.ended
                );
                if (anyPlaying && el.paused) show = false;
            }
            btn.style.display = show ? "block" : "none";
            if (show) updatePos();
        });
    }, { threshold: 0.5 }).observe(el);

    // Also update button visibility when play/pause state changes
    if (IS_SOCIAL && el.tagName === "VIDEO") {
        el.addEventListener("play", () => { btn.style.display = "block"; updatePos(); });
        el.addEventListener("pause", () => {
            const anyPlaying = [...document.querySelectorAll("video")].some(
                v => v !== el && !v.paused && !v.ended
            );
            if (anyPlaying) btn.style.display = "none";
        });
    }

    document.addEventListener("scroll", updatePos, { passive: true });
    window.addEventListener("resize", updatePos, { passive: true });
    updatePos();
}

/* ─── MutationObserver ──────────────────────────────────────────────────── */
function processNode(node) {
    if (!(node instanceof Element)) return;
    if (["IMG", "VIDEO", "AUDIO"].includes(node.tagName)) attach(node);
    node.querySelectorAll("img,video,audio").forEach(attach);
}

new MutationObserver(muts => {
    muts.forEach(m => {
        m.addedNodes.forEach(processNode);
        if (m.type === "attributes") processNode(m.target);
    });
}).observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["src", "data-video-url", "data-src"],
});

document.querySelectorAll("img,video,audio").forEach(attach);

window.addEventListener("load", () => {
    document.querySelectorAll("img,video,audio").forEach(attach);
});

/* ─── Context menu trigger ───────────────────────────────────────────────── */
chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === "SCAN_CONTEXT") {
        const el = [...document.querySelectorAll("img,video,audio")]
            .find(e => getRealUrl(e) === msg.url || e.src === msg.url);
        if (el) scan(el);
    }
});
