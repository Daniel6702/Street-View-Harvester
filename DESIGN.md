# Street View Harvest Monitor Design System

## 1. Atmosphere & Identity

The monitor is a quiet operations readout: a small, dependable window into one
running harvest. Its signature is an honest status line paired with a compact
grid of scalar facts, so the page is useful over a terminal tunnel without
competing with the harvest itself.

## 2. Color

### Palette

| Role | Token | Light | Dark | Usage |
|------|-------|-------|------|-------|
| Surface/primary | --surface-primary | #FFFFFF | #111111 | Page canvas |
| Surface/secondary | --surface-secondary | #F7F6F3 | #1D1D1B | Status panel |
| Text/primary | --text-primary | #111111 | #F7F6F3 | Headings and values |
| Text/secondary | --text-secondary | #5F5E5A | #B8B6AE | Supporting copy and labels |
| Border/default | --border-default | #EAEAEA | #383835 | Panel and grid dividers |
| Accent/primary | --accent-primary | #1F6C9F | #8CC7EE | Links and focus |
| Status/running | --status-running | #346538 | #A9D5A6 | Running indicator |
| Status/complete | --status-complete | #1F6C9F | #8CC7EE | Complete indicator |
| Status/error | --status-error | #9F2F2D | #F3AAA8 | Error indicator |

### Rules

- Use semantic tokens only; do not add decorative color.
- Prefer the light palette and rely on borders plus tonal surfaces for depth.
- Status color supplements, but never replaces, the written state.

## 3. Typography

### Scale

| Level | Size | Weight | Line Height | Usage |
|-------|------|--------|-------------|-------|
| H1 | 28px / 1.75rem | 700 | 1.3 | Page title |
| H2 | 18px / 1.125rem | 600 | 1.4 | Status heading |
| Body | 16px / 1rem | 400 | 1.6 | Supporting text |
| Body/sm | 14px / 0.875rem | 400 | 1.5 | Values and labels |
| Caption | 12px / 0.75rem | 600 | 1.4 | Eyebrow and update time |

### Font Stack

- Primary: `"Helvetica Neue", "Segoe UI", sans-serif`
- Mono: `ui-monospace, SFMono-Regular, Menlo, monospace`

### Rules

- Use the mono stack for scalar values so changing numbers are easy to scan.
- Body text stays at or above 14px.

## 4. Spacing & Layout

### Base Unit

All spacing derives from a base of **4px**.

| Token | Value | Usage |
|-------|-------|-------|
| --space-1 | 4px | Label/value separation |
| --space-2 | 8px | Compact grid gaps |
| --space-3 | 12px | Panel inner rhythm |
| --space-4 | 16px | Standard page padding |
| --space-6 | 24px | Panel padding |
| --space-8 | 32px | Page section separation |

### Grid

- Max content width: 720px.
- One-column page shell with a two-column fact grid at 640px and above.
- Mobile breakpoint: 640px; the page remains readable without horizontal scroll.

### Rules

- Use the base-4 scale for every margin, padding, and gap.
- Keep the status page intentionally narrow for SSH-forwarded terminal work.

## 5. Components

### Status panel

- **Structure**: `main` > `header` + `section` containing a heading and a `dl` with progress, pace, and update facts.
- **Variants**: running, complete, stopped.
- **Spacing**: `--space-3`, `--space-4`, `--space-6`, `--space-8`.
- **States**: static readout; values update through same-origin polling, including coordinator-derived pace in items per second.
- **Accessibility**: semantic headings, `dl` labels, visible focus, text state.
- **Motion**: no essential motion; updates do not animate layout.

## 6. Motion & Interaction

The page has no controls and no essential animation. It refreshes its scalar
readout from the same-origin progress endpoint at a quiet interval. Respect
`prefers-reduced-motion` if motion is ever added; never animate layout.

## 7. Depth & Surface

### Strategy

Use **borders-only**. The page canvas and status panel use the surface tokens;
the panel and fact cells are separated with the default border token. No
shadows, gradients, images, or remote assets are part of this surface.
