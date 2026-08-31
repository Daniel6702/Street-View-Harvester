import html
from typing import Final


_PAGE_TITLE: Final = html.escape("Street View Harvest Monitor", quote=True)
HTML_PAGE: Final = (
    "<!doctype html>"
    '<html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1">'
    f"<title>{_PAGE_TITLE}</title>"
    "<style>"
    ":root{--surface-primary:#fff;--surface-secondary:#f7f6f3;--text-primary:#111;"
    "--text-secondary:#5f5e5a;--border-default:#eaeaea;--accent-primary:#1f6c9f;"
    "--status-running:#346538;--status-complete:#1f6c9f;--status-error:#9f2f2d;"
    "--space-1:4px;--space-2:8px;--space-3:12px;--space-4:16px;--space-6:24px;--space-8:32px}"
    "*{box-sizing:border-box}body{margin:0;background:var(--surface-primary);color:var(--text-primary);"
    "font-family:\"Helvetica Neue\",\"Segoe UI\",sans-serif;font-size:16px;line-height:1.6}"
    ".page{max-width:720px;margin:0 auto;padding:var(--space-8) var(--space-4)}"
    "header{margin-bottom:var(--space-8)}.eyebrow{margin:0 0 var(--space-2);color:var(--text-secondary);"
    "font-size:12px;font-weight:600;letter-spacing:.08em;text-transform:uppercase}"
    "h1{margin:0;font-size:28px;line-height:1.3}h2{margin:0;font-size:18px;line-height:1.4}"
    ".panel{border:1px solid var(--border-default);background:var(--surface-secondary);padding:var(--space-6)}"
    ".state{margin:var(--space-2) 0 0;color:var(--status-running);font-size:14px;font-weight:600}"
    "body[data-state=complete] .state{color:var(--status-complete)}"
    "body[data-state=stopped] .state,body[data-state=error] .state{color:var(--status-error)}"
    ".stats{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));margin:var(--space-4) 0 0}"
    ".stat{border-top:1px solid var(--border-default);padding:var(--space-3) var(--space-2) 0 0}"
    "dt{color:var(--text-secondary);font-size:12px;font-weight:600;letter-spacing:.02em}"
    "dd{margin:var(--space-1) 0 0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:14px}"
    ".notice{min-height:24px;margin:var(--space-4) 0 0;color:var(--text-secondary);font-size:14px}"
    "@media (max-width:639px){.page{padding:var(--space-6) var(--space-4)}header{margin-bottom:var(--space-6)}}"
    "@media (prefers-color-scheme:dark){:root{--surface-primary:#111;--surface-secondary:#1d1d1b;"
    "--text-primary:#f7f6f3;--text-secondary:#b8b6ae;--border-default:#383835;"
    "--accent-primary:#8cc7ee;--status-running:#a9d5a6;--status-complete:#8cc7ee;--status-error:#f3aaa8}}"
    '</style></head><body data-state="running"><main class="page">'
    '<header><p class="eyebrow">Street View harvest</p><h1>Harvest monitor</h1>'
    '<p id="state" class="state" aria-live="polite">Loading status</p></header>'
    '<section class="panel" aria-labelledby="status-heading"><h2 id="status-heading">Current status</h2>'
    '<dl class="stats"><div class="stat"><dt>Current / target</dt><dd><span id="current">-</span> / <span id="target">-</span></dd></div>'
    '<div class="stat"><dt>Added</dt><dd id="added">-</dd></div>'
    '<div class="stat"><dt>Queries</dt><dd id="queries">-</dd></div>'
    '<div class="stat"><dt>Queries / second</dt><dd id="queries-per-second">- queries/s</dd></div>'
    '<div class="stat"><dt>Images / second</dt><dd id="images-per-second">- images/s</dd></div>'
    '<div class="stat"><dt>Estimated finish</dt><dd id="estimated-finish">Estimating</dd></div>'
    '<div class="stat"><dt>Unit</dt><dd id="unit">-</dd></div>'
    '<div class="stat"><dt>Last update</dt><dd id="last-update">-</dd></div></dl>'
    '<p id="notice" class="notice" role="status"></p></section></main>'
    '<script>const text=(id,value)=>{document.getElementById(id).textContent=value};'
    'const rate=(value,unit)=>value===null?"Not downloading":`${value.toFixed(2)} ${unit}/s`;'
    'const finish=status=>status.state==="complete"?"Complete":status.estimated_finish_at===null?"Estimating":new Date(status.estimated_finish_at*1000).toLocaleString();'
    'const refresh=async()=>{try{const response=await fetch("/api/v1/progress",{cache:"no-store"});'
    'if(!response.ok)throw new Error("status unavailable");const status=await response.json();'
    'document.body.dataset.state=status.state;text("state",status.state);text("current",status.current);'
    'text("target",status.target);text("added",status.added);text("queries",status.queries);'
    'text("queries-per-second",rate(status.queries_per_second,"queries"));'
    'text("images-per-second",rate(status.images_per_second,"images"));'
    'text("estimated-finish",finish(status));text("unit",status.unit);'
    'text("last-update",new Date(status.last_update*1000).toLocaleString());text("notice","");'
    '}catch(error){text("notice","Status temporarily unavailable; retrying.")}};refresh();setInterval(refresh,2000);</script>'
    "</body></html>"
)
