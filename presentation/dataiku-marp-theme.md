---
marp: true
theme: default
paginate: true
---

<!--
  ╔══════════════════════════════════════════════════════════════╗
  ║         DATAIKU MARP TEMPLATE  ·  Brand Guidelines 2026     ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  CORE COLORS                                                 ║
  ║  Core Black        #0F1314    PMS Black 6 C                  ║
  ║  Core White        #FFFFFA    PMS 1-1 C                      ║
  ║                                                              ║
  ║  GREENS                                                      ║
  ║  Dark Green        #0A2A1F    PMS 627 C  — primary dark BG   ║
  ║  Green             #63FF91    PMS 3385 C — primary accent    ║
  ║  Light Green       #D1FFE6    PMS 331 C  — light tint        ║
  ║                                                              ║
  ║  BLUES & GREYS                                               ║
  ║  Blue              #7099FF    — secondary accent             ║
  ║  Dark Grey         #1D2220    — near-black alt               ║
  ║  Dark Blue Grey    #081030    — deepest dark surface         ║
  ║  Blue Grey         #2B3B64    — muted dark accent            ║
  ║  Light Blue Grey   #ABCFFA    — light dividers, muted text   ║
  ║                                                              ║
  ║  TYPOGRAPHY                                                  ║
  ║  Display / H1   DM Serif Display — editorial weight          ║
  ║  Headings       DM Sans Bold — clean, modern                 ║
  ║  Body           DM Sans Regular                              ║
  ║  Code / Mono    DM Mono                                      ║
  ║                                                              ║
  ║  USAGE                                                       ║
  ║  marp --html --pdf dataiku-marp-theme.md                     ║
  ║                                                              ║
  ║  LOGO                                                        ║
  ║  575 Lab condensed logo — embedded as base64 data URIs       ║
  ║  White variant: cover, section, dark, closing slides         ║
  ║  Black variant: default (light) slides                       ║
  ╚══════════════════════════════════════════════════════════════╝

  HOW TO USE SLIDE CLASSES  (add to the front-matter of each slide)
  ─────────────────────────────────────────────────────────────────
  _class: cover          → dark green hero slide (title / cover)
  _class: section        → bright green section divider
  _class: dark           → dark blue-grey content slide
  _class: light          → off-white content slide (default)
  _class: closing        → gradient closing / thank-you slide
-->

<style>
/* ─── Google Fonts ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&display=swap');

/* ─── Brand Tokens ─────────────────────────────────────────────── */
:root {
  --core-black:       #0F1314;
  --core-white:       #FFFFFA;
  --dark-green:       #0A2A1F;
  --green:            #63FF91;
  --light-green:      #D1FFE6;
  --blue:             #7099FF;
  --dark-grey:        #1D2220;
  --dark-blue-grey:   #081030;
  --blue-grey:        #2B3B64;
  --light-blue-grey:  #ABCFFA;

  --ink:              var(--core-black);
  --ink-soft:         #4A5A55;        /* derived muted body text   */
  --offwhite-bg:      #F4F7F2;        /* slightly cool off-white   */

  --font-display:     'DM Serif Display', Georgia, serif;
  --font-sans:        'DM Sans', system-ui, sans-serif;
  --font-mono:        'DM Mono', 'Courier New', monospace;

  --radius:           12px;
  --gap:              2rem;

  --logo-white:       url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjc4IiBoZWlnaHQ9IjU2IiB2aWV3Qm94PSIwIDAgMjc4IDU2IiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNNTIuOTU4OSAzOC42NzcySDMwLjA0NDZWNDIuOTYwN0g1Mi45NTg5VjM4LjY3NzJaIiBmaWxsPSIjRkZGRUY5Ii8+CjxwYXRoIGQ9Ik01MC44MjE2IDMuODM0OEM0OS42MzA0IDEuNjYzNyA0Ny4zMjQzIDAuMTg0OTk4IDQ0LjY2NjEgMC4xODQ5OThDNDAuNzkzMyAwLjE4NDk5OCAzNy42NTM5IDMuMzI0MyAzNy42NTM5IDcuMTk3MDhDMzcuNjUzOSA3LjU2Njc2IDM3LjY4OTEgNy45MjQ3IDM3Ljc0NzggOC4yNzY3N0wzNy4xNTUxIDkuMDA0MzhMMC4xMTY1NiA1NC42OTc0Qy0wLjA2NTM0NjQgNTQuOTIwMyAtMC4wMzAxMzg2IDU1LjI0ODkgMC4xOTI4NDQgNTUuNDMwOEMwLjM5ODIyMyA1NS41OTUxIDAuNjk3NDg5IDU1LjU4MzQgMC44ODUyNjMgNTUuMzk1NkwxNi40NzY0IDM5LjgyMjRDMTkuMzUxNyAzNi45NTMgMjMuMjQ4MSAzNS4zMzkzIDI3LjMxNDYgMzUuMzM5M0gzMi43NDI0QzQ0LjYwMTYgMzUuMzM5MyA1MS43MzEyIDI4LjU4NTQgNTAuNDQwMiAxNC43MDc5QzQ5Ljk5NDIgOS45MzE1IDUwLjI0MDcgNy45MTg4MyA1MS44Nzc5IDUuOTAwMjlDNTIuNzIyOCA0Ljg2MTY4IDU0LjM2IDIuODMxNCA1NC4zNiAyLjgzMTRMNTIuNDQxMiAzLjM3MTI0TDUwLjgxNTggMy44Mjg5M0w1MC44MjE2IDMuODM0OFpNNDQuODEyOCA4LjA1OTY2QzQzLjc2MjUgOC4wNTk2NiA0Mi45MTE2IDcuMjA4ODIgNDIuOTExNiA2LjE1ODQ3QzQyLjkxMTYgNS4xMDgxMyA0My43NjI1IDQuMjU3MjkgNDQuODEyOCA0LjI1NzI5QzQ1Ljg2MzIgNC4yNTcyOSA0Ni43MTQgNS4xMDgxMyA0Ni43MTQgNi4xNTg0N0M0Ni43MTQgNy4yMDg4MiA0NS44NjMyIDguMDU5NjYgNDQuODEyOCA4LjA1OTY2WiIgZmlsbD0iI0ZGRkVGOSIvPgo8cGF0aCBkPSJNNzMuMTk1IDIyLjI0M0w3NS43MzIgLTQuNDEwNzRlLTA2SDk3LjYyMVY0LjM2Nkg3OS41MDhMNzcuNzk3IDE4LjU4NUM3OS42MjYgMTYuNzU2IDgyLjYzNSAxNC45ODYgODcuMTE5IDE0Ljk4NkM5NC44NDggMTQuOTg2IDEwMC4wOTkgMjAuMjM3IDEwMC4wOTkgMjguNjE1QzEwMC4wOTkgMzYuOTM0IDk0LjU1MyA0My4wNyA4NS44MjEgNDMuMDdDNzguMDMzIDQzLjA3IDcyLjU0NiAzOS4xMTcgNzEuNDI1IDMwLjYyMUg3Ni42MTdDNzcuMDg5IDM1LjEwNSA3OS44NjIgMzkuMDU4IDg1LjcwMyAzOS4wNThDOTEuNjAzIDM5LjA1OCA5NC45NjYgMzQuNzUxIDk0Ljk2NiAyOC44NTFDOTQuOTY2IDIzLjM2NCA5MS42MDMgMTguOTM5IDg1LjkzOSAxOC45MzlDODEuNzUgMTguOTM5IDc5LjI3MiAyMS4wMDQgNzguMDMzIDIzLjA2OUw3My4xOTUgMjIuMjQzWk0xMTQuNjQ3IDQyLjE4NUgxMDguOTI0QzExMS42OTcgMzUuMDQ2IDExNS4wMDEgMjguMzc5IDExOC4xMjggMjEuNDc2QzEyMC4zNyAxNi41MiAxMjQuMjY0IDguOTY4IDEyNi43NDIgNC4yNDhWNC4xODg5OUMxMjUuNjIxIDQuMjQ4IDEyNC42MTggNC4yNDggMTIzLjQ5NyA0LjI0OEgxMDUuMTQ4Vi00LjQxMDc0ZS0wNkgxMzEuMTA4VjMuODkzOTlMMTI5LjY5MiA3LjAyMUMxMjMuNTU2IDIwLjUzMiAxMTkuMDcyIDMwLjE0OSAxMTQuNjQ3IDQyLjE4NVpNMTM2Ljg2NCAyMi4yNDNMMTM5LjQwMSAtNC40MTA3NGUtMDZIMTYxLjI5VjQuMzY2SDE0My4xNzdMMTQxLjQ2NiAxOC41ODVDMTQzLjI5NSAxNi43NTYgMTQ2LjMwNCAxNC45ODYgMTUwLjc4OCAxNC45ODZDMTU4LjUxNyAxNC45ODYgMTYzLjc2OCAyMC4yMzcgMTYzLjc2OCAyOC42MTVDMTYzLjc2OCAzNi45MzQgMTU4LjIyMiA0My4wNyAxNDkuNDkgNDMuMDdDMTQxLjcwMiA0My4wNyAxMzYuMjE1IDM5LjExNyAxMzUuMDk0IDMwLjYyMUgxNDAuMjg2QzE0MC43NTggMzUuMTA1IDE0My41MzEgMzkuMDU4IDE0OS4zNzIgMzkuMDU4QzE1NS4yNzIgMzkuMDU4IDE1OC42MzUgMzQuNzUxIDE1OC42MzUgMjguODUxQzE1OC42MzUgMjMuMzY0IDE1NS4yNzIgMTguOTM5IDE0OS42MDggMTguOTM5QzE0NS40MTkgMTguOTM5IDE0Mi45NDEgMjEuMDA0IDE0MS43MDIgMjMuMDY5TDEzNi44NjQgMjIuMjQzWk0xODQuMTUgLTQuNDEwNzRlLTA2SDE4OS4xNjVWMzcuODE5SDIwOC4xNjNWNDIuMTg1SDE4NC4xNVYtNC40MTA3NGUtMDZaTTIxNy4xNDkgMjAuNTkxSDIxMi40ODhDMjEyLjQ4OCAxNC44MDkgMjE3LjUwMyAxMC4zODQgMjI0Ljk5NiAxMC4zODRDMjMzLjAyIDEwLjM4NCAyMzYuOTczIDE0LjE2IDIzNi45NzMgMjMuNzc3QzIzNi45NzMgMjUuMDE2IDIzNi45MTQgMjcuMDIyIDIzNi45MTQgMjguNDM4VjM2LjA0OUMyMzYuOTE0IDM4LjY0NSAyMzcuMDkxIDQwLjc2OSAyMzcuNTA0IDQyLjE4NUgyMzMuMDJDMjMyLjc4NCA0MC44MjggMjMyLjYwNyAzOS40NzEgMjMyLjYwNyAzNy40MDZDMjMwLjY2IDQwLjUzMyAyMjcuNTMzIDQyLjc3NSAyMjEuOTI4IDQyLjc3NUMyMTUuNTU2IDQyLjc3NSAyMTEuMDcyIDM5LjUzIDIxMS4wNzIgMzMuODY2QzIxMS4wNzIgMjcuNzg5IDIxNi42MTggMjUuNzI0IDIyNC4xNyAyNC40ODVDMjI2LjcwNyAyNC4wNzIgMjI5Ljc3NSAyMy42NTkgMjMyLjMxMiAyMy40ODJDMjMyLjMxMiAxNi44NzQgMjMwLjAxMSAxNC4yMTkgMjI0Ljc2IDE0LjIxOUMyMTkuODA0IDE0LjIxOSAyMTcuNTYyIDE2LjU3OSAyMTcuMTQ5IDIwLjU5MVpNMjMyLjM3MSAyOC40MzhDMjMyLjM3MSAyNy42MTIgMjMyLjM3MSAyNy4wODEgMjMyLjQzIDI2LjkwNEwyMjkuODM0IDI3LjI1OEMyMjAuOTg0IDI4LjQ5NyAyMTYuMjA1IDI5LjM4MiAyMTYuMjA1IDMzLjgwN0MyMTYuMjA1IDM3LjM0NyAyMTguODYgMzkuMzUzIDIyMy4xMDggMzkuMzUzQzIyOS4wNjcgMzkuMzUzIDIzMi4zNzEgMzUuNTc3IDIzMi4zNzEgMjguNDM4Wk0yNDQuNDU3IDQyLjE4NVYtNC40MTA3NGUtMDZIMjQ5LjM1NFY5LjAyN0MyNDkuMzU0IDExLjA5MiAyNDkuMzU0IDEzLjg2NSAyNDkuMjk1IDE1Ljk4OUMyNTEuMDA2IDEzLjE1NyAyNTQuNjY0IDEwLjM4NCAyNjAuMTUxIDEwLjM4NEMyNjguNTg4IDEwLjM4NCAyNzQuMzExIDE2LjkzMyAyNzQuMzExIDI2LjQ5MUMyNzQuMzExIDM2LjQwMyAyNjguMjM0IDQyLjk1MiAyNTkuNzk3IDQyLjk1MkMyNTQuNDI4IDQyLjk1MiAyNTAuODI5IDQwLjIzOCAyNDkgMzcuNDY1VjQyLjE4NUgyNDQuNDU3Wk0yNTkuMDg5IDM4Ljk5OUMyNjUuNDYxIDM4Ljk5OSAyNjkuMDAxIDM0LjIyIDI2OS4wMDEgMjYuNjY4QzI2OS4wMDEgMTkuNDcgMjY1LjQ2MSAxNC4zMzcgMjU5LjA4OSAxNC4zMzdDMjUyLjcxNyAxNC4zMzcgMjQ5LjIzNiAxOS40NyAyNDkuMjM2IDI2LjY2OEMyNDkuMjM2IDM0LjIyIDI1Mi42NTggMzguOTk5IDI1OS4wODkgMzguOTk5WiIgZmlsbD0iI0ZGRkVGOSIvPgo8L3N2Zz4K');
  --logo-black:       url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjc4IiBoZWlnaHQ9IjU2IiB2aWV3Qm94PSIwIDAgMjc4IDU2IiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNNTIuOTU4OSAzOC42NzcySDMwLjA0NDZWNDIuOTYwN0g1Mi45NTg5VjM4LjY3NzJaIiBmaWxsPSIjMUExQTFBIi8+CjxwYXRoIGQ9Ik01MC44MjE2IDMuODM0OEM0OS42MzA0IDEuNjYzNyA0Ny4zMjQzIDAuMTg0OTk4IDQ0LjY2NjEgMC4xODQ5OThDNDAuNzkzMyAwLjE4NDk5OCAzNy42NTM5IDMuMzI0MyAzNy42NTM5IDcuMTk3MDhDMzcuNjUzOSA3LjU2Njc2IDM3LjY4OTEgNy45MjQ3IDM3Ljc0NzggOC4yNzY3N0wzNy4xNTUxIDkuMDA0MzhMMC4xMTY1NiA1NC42OTc0Qy0wLjA2NTM0NjQgNTQuOTIwMyAtMC4wMzAxMzg2IDU1LjI0ODkgMC4xOTI4NDQgNTUuNDMwOEMwLjM5ODIyMyA1NS41OTUxIDAuNjk3NDg5IDU1LjU4MzQgMC44ODUyNjMgNTUuMzk1NkwxNi40NzY0IDM5LjgyMjRDMTkuMzUxNyAzNi45NTMgMjMuMjQ4MSAzNS4zMzkzIDI3LjMxNDYgMzUuMzM5M0gzMi43NDI0QzQ0LjYwMTYgMzUuMzM5MyA1MS43MzEyIDI4LjU4NTQgNTAuNDQwMiAxNC43MDc5QzQ5Ljk5NDIgOS45MzE1IDUwLjI0MDcgNy45MTg4MyA1MS44Nzc5IDUuOTAwMjlDNTIuNzIyOCA0Ljg2MTY4IDU0LjM2IDIuODMxNCA1NC4zNiAyLjgzMTRMNTIuNDQxMiAzLjM3MTI0TDUwLjgxNTggMy44Mjg5M0w1MC44MjE2IDMuODM0OFpNNDQuODEyOCA4LjA1OTY2QzQzLjc2MjUgOC4wNTk2NiA0Mi45MTE2IDcuMjA4ODIgNDIuOTExNiA2LjE1ODQ3QzQyLjkxMTYgNS4xMDgxMyA0My43NjI1IDQuMjU3MjkgNDQuODEyOCA0LjI1NzI5QzQ1Ljg2MzIgNC4yNTcyOSA0Ni43MTQgNS4xMDgxMyA0Ni43MTQgNi4xNTg0N0M0Ni43MTQgNy4yMDg4MiA0NS44NjMyIDguMDU5NjYgNDQuODEyOCA4LjA1OTY2WiIgZmlsbD0iIzFBMUExQSIvPgo8cGF0aCBkPSJNNzMuMTk1IDIyLjI0M0w3NS43MzIgLTQuNDEwNzRlLTA2SDk3LjYyMVY0LjM2Nkg3OS41MDhMNzcuNzk3IDE4LjU4NUM3OS42MjYgMTYuNzU2IDgyLjYzNSAxNC45ODYgODcuMTE5IDE0Ljk4NkM5NC44NDggMTQuOTg2IDEwMC4wOTkgMjAuMjM3IDEwMC4wOTkgMjguNjE1QzEwMC4wOTkgMzYuOTM0IDk0LjU1MyA0My4wNyA4NS44MjEgNDMuMDdDNzguMDMzIDQzLjA3IDcyLjU0NiAzOS4xMTcgNzEuNDI1IDMwLjYyMUg3Ni42MTdDNzcuMDg5IDM1LjEwNSA3OS44NjIgMzkuMDU4IDg1LjcwMyAzOS4wNThDOTEuNjAzIDM5LjA1OCA5NC45NjYgMzQuNzUxIDk0Ljk2NiAyOC44NTFDOTQuOTY2IDIzLjM2NCA5MS42MDMgMTguOTM5IDg1LjkzOSAxOC45MzlDODEuNzUgMTguOTM5IDc5LjI3MiAyMS4wMDQgNzguMDMzIDIzLjA2OUw3My4xOTUgMjIuMjQzWk0xMTQuNjQ3IDQyLjE4NUgxMDguOTI0QzExMS42OTcgMzUuMDQ2IDExNS4wMDEgMjguMzc5IDExOC4xMjggMjEuNDc2QzEyMC4zNyAxNi41MiAxMjQuMjY0IDguOTY4IDEyNi43NDIgNC4yNDhWNC4xODg5OUMxMjUuNjIxIDQuMjQ4IDEyNC42MTggNC4yNDggMTIzLjQ5NyA0LjI0OEgxMDUuMTQ4Vi00LjQxMDc0ZS0wNkgxMzEuMTA4VjMuODkzOTlMMTI5LjY5MiA3LjAyMUMxMjMuNTU2IDIwLjUzMiAxMTkuMDcyIDMwLjE0OSAxMTQuNjQ3IDQyLjE4NVpNMTM2Ljg2NCAyMi4yNDNMMTM5LjQwMSAtNC40MTA3NGUtMDZIMTYxLjI5VjQuMzY2SDE0My4xNzdMMTQxLjQ2NiAxOC41ODVDMTQzLjI5NSAxNi43NTYgMTQ2LjMwNCAxNC45ODYgMTUwLjc4OCAxNC45ODZDMTU4LjUxNyAxNC45ODYgMTYzLjc2OCAyMC4yMzcgMTYzLjc2OCAyOC42MTVDMTYzLjc2OCAzNi45MzQgMTU4LjIyMiA0My4wNyAxNDkuNDkgNDMuMDdDMTQxLjcwMiA0My4wNyAxMzYuMjE1IDM5LjExNyAxMzUuMDk0IDMwLjYyMUgxNDAuMjg2QzE0MC43NTggMzUuMTA1IDE0My41MzEgMzkuMDU4IDE0OS4zNzIgMzkuMDU4QzE1NS4yNzIgMzkuMDU4IDE1OC42MzUgMzQuNzUxIDE1OC42MzUgMjguODUxQzE1OC42MzUgMjMuMzY0IDE1NS4yNzIgMTguOTM5IDE0OS42MDggMTguOTM5QzE0NS40MTkgMTguOTM5IDE0Mi45NDEgMjEuMDA0IDE0MS43MDIgMjMuMDY5TDEzNi44NjQgMjIuMjQzWk0xODQuMTUgLTQuNDEwNzRlLTA2SDE4OS4xNjVWMzcuODE5SDIwOC4xNjNWNDIuMTg1SDE4NC4xNVYtNC40MTA3NGUtMDZaTTIxNy4xNDkgMjAuNTkxSDIxMi40ODhDMjEyLjQ4OCAxNC44MDkgMjE3LjUwMyAxMC4zODQgMjI0Ljk5NiAxMC4zODRDMjMzLjAyIDEwLjM4NCAyMzYuOTczIDE0LjE2IDIzNi45NzMgMjMuNzc3QzIzNi45NzMgMjUuMDE2IDIzNi45MTQgMjcuMDIyIDIzNi45MTQgMjguNDM4VjM2LjA0OUMyMzYuOTE0IDM4LjY0NSAyMzcuMDkxIDQwLjc2OSAyMzcuNTA0IDQyLjE4NUgyMzMuMDJDMjMyLjc4NCA0MC44MjggMjMyLjYwNyAzOS40NzEgMjMyLjYwNyAzNy40MDZDMjMwLjY2IDQwLjUzMyAyMjcuNTMzIDQyLjc3NSAyMjEuOTI4IDQyLjc3NUMyMTUuNTU2IDQyLjc3NSAyMTEuMDcyIDM5LjUzIDIxMS4wNzIgMzMuODY2QzIxMS4wNzIgMjcuNzg5IDIxNi42MTggMjUuNzI0IDIyNC4xNyAyNC40ODVDMjI2LjcwNyAyNC4wNzIgMjI5Ljc3NSAyMy42NTkgMjMyLjMxMiAyMy40ODJDMjMyLjMxMiAxNi44NzQgMjMwLjAxMSAxNC4yMTkgMjI0Ljc2IDE0LjIxOUMyMTkuODA0IDE0LjIxOSAyMTcuNTYyIDE2LjU3OSAyMTcuMTQ5IDIwLjU5MVpNMjMyLjM3MSAyOC40MzhDMjMyLjM3MSAyNy42MTIgMjMyLjM3MSAyNy4wODEgMjMyLjQzIDI2LjkwNEwyMjkuODM0IDI3LjI1OEMyMjAuOTg0IDI4LjQ5NyAyMTYuMjA1IDI5LjM4MiAyMTYuMjA1IDMzLjgwN0MyMTYuMjA1IDM3LjM0NyAyMTguODYgMzkuMzUzIDIyMy4xMDggMzkuMzUzQzIyOS4wNjcgMzkuMzUzIDIzMi4zNzEgMzUuNTc3IDIzMi4zNzEgMjguNDM4Wk0yNDQuNDU3IDQyLjE4NVYtNC40MTA3NGUtMDZIMjQ5LjM1NFY5LjAyN0MyNDkuMzU0IDExLjA5MiAyNDkuMzU0IDEzLjg2NSAyNDkuMjk1IDE1Ljk4OUMyNTEuMDA2IDEzLjE1NyAyNTQuNjY0IDEwLjM4NCAyNjAuMTUxIDEwLjM4NEMyNjguNTg4IDEwLjM4NCAyNzQuMzExIDE2LjkzMyAyNzQuMzExIDI2LjQ5MUMyNzQuMzExIDM2LjQwMyAyNjguMjM0IDQyLjk1MiAyNTkuNzk3IDQyLjk1MkMyNTQuNDI4IDQyLjk1MiAyNTAuODI5IDQwLjIzOCAyNDkgMzcuNDY1VjQyLjE4NUgyNDQuNDU3Wk0yNTkuMDg5IDM4Ljk5OUMyNjUuNDYxIDM4Ljk5OSAyNjkuMDAxIDM0LjIyIDI2OS4wMDEgMjYuNjY4QzI2OS4wMDEgMTkuNDcgMjY1LjQ2MSAxNC4zMzcgMjU5LjA4OSAxNC4zMzdDMjUyLjcxNyAxNC4zMzcgMjQ5LjIzNiAxOS40NyAyNDkuMjM2IDI2LjY2OEMyNDkuMjM2IDM0LjIyIDI1Mi42NTggMzguOTk5IDI1OS4wODkgMzguOTk5WiIgZmlsbD0iIzFBMUExQSIvPgo8L3N2Zz4K');
}

/* ─── Global Reset ─────────────────────────────────────────────── */
section {
  font-family: var(--font-sans);
  font-size:   20px;
  line-height: 1.6;
  color:       var(--ink);
  background:  var(--offwhite-bg);
  padding:     56px 72px;
  box-sizing:  border-box;
  width:  1280px;
  height: 720px;
  position: relative;
  overflow: hidden;
}

/* Corner accent — a small green bar in the top-right */
section::before {
  content:  '';
  position: absolute;
  top:    0;
  right:  0;
  width:  120px;
  height: 8px;
  background: var(--green);
}

/* Page number styling */
section::after {
  font-family: var(--font-sans);
  font-size:   13px;
  font-weight: 500;
  color:       var(--blue-grey);
  bottom:      28px;
  right:       72px;
}

/* ─── Typography ───────────────────────────────────────────────── */
h1 {
  font-family: var(--font-display);
  font-size:   3.4rem;
  line-height: 1.1;
  color:       var(--dark-green);
  margin:      0 0 0.4em;
  letter-spacing: -0.02em;
}

h2 {
  font-family: var(--font-sans);
  font-size:   1.9rem;
  font-weight: 700;
  color:       var(--dark-green);
  margin:      0 0 0.5em;
  letter-spacing: -0.01em;
}

h3 {
  font-family: var(--font-sans);
  font-size:   1.2rem;
  font-weight: 600;
  color:       var(--blue-grey);
  margin:      0 0 0.35em;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

h4 {
  font-family: var(--font-sans);
  font-size:   1rem;
  font-weight: 600;
  color:       var(--ink-soft);
  margin:      0 0 0.25em;
}

p { margin: 0 0 1em; }

strong { color: var(--dark-green); font-weight: 700; }
em     { color: var(--blue-grey); font-style: italic; }

a {
  color:           var(--blue-grey);
  text-decoration: none;
  border-bottom:   1px solid var(--green);
}

/* ─── Lists ────────────────────────────────────────────────────── */
ul, ol {
  padding-left: 1.4em;
  margin: 0 0 1em;
}

li { margin-bottom: 0.45em; }

li::marker { color: var(--dark-green); }

/* ─── Code ─────────────────────────────────────────────────────── */
code {
  font-family:      var(--font-mono);
  font-size:        0.82em;
  background:       rgba(10,42,31,.08);
  color:            var(--dark-green);
  padding:          0.15em 0.4em;
  border-radius:    4px;
}

pre {
  background:    var(--dark-blue-grey);
  color:         var(--light-green);
  border-radius: var(--radius);
  padding:       1.25em 1.5em;
  font-size:     0.78em;
  overflow:      hidden;
}

pre code {
  background: transparent;
  color:       inherit;
  padding:     0;
}

/* ─── Tables ───────────────────────────────────────────────────── */
table {
  width:           100%;
  border-collapse: collapse;
  font-size:       0.88em;
}

thead tr {
  background:  var(--dark-green);
  color:       var(--core-white);
}

th {
  padding:     0.6em 1em;
  text-align:  left;
  font-weight: 600;
  letter-spacing: 0.03em;
}

td {
  padding:       0.55em 1em;
  border-bottom: 1px solid rgba(43,59,100,.2);
}

tr:nth-child(even) td {
  background: rgba(10,42,31,.04);
}

/* ─── Blockquote ───────────────────────────────────────────────── */
blockquote {
  border-left: 4px solid var(--green);
  padding:     0.5em 1.5em;
  margin:      1em 0;
  background:  rgba(99,255,145,.10);
  border-radius: 0 var(--radius) var(--radius) 0;
  font-style:  italic;
  color:       var(--ink-soft);
}

blockquote p { margin: 0; }

/* ─── Horizontal Rule ──────────────────────────────────────────── */
hr {
  border: none;
  height: 2px;
  background: linear-gradient(90deg, var(--green) 0%, var(--blue) 60%, transparent 100%);
  margin: 1.5em 0;
}

/* ─── Images ───────────────────────────────────────────────────── */
img {
  border-radius: var(--radius);
  max-width:     100%;
}


/* ─── Logo (575 Lab) ───────────────────────────────────────────── */
.logo {
  position: absolute;
  top:      40px;
  left:     72px;
  width:    140px;
  height:   28px;
  background-image:    var(--logo-black);
  background-repeat:   no-repeat;
  background-size:     contain;
  background-position: left center;
}

section.cover    .logo,
section.section  .logo,
section.dark     .logo,
section.closing  .logo {
  background-image: var(--logo-white);
}

/* Center the logo on closing slide */
section.closing .logo {
  top:     40px;
  left:    50%;
  transform: translateX(-50%);
  background-position: center;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   SLIDE VARIANTS
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

/* ── COVER  (dark green, large display type) ────────────────────── */
section.cover {
  background: var(--dark-green);
  color:      var(--core-white);
  padding:    80px 96px;
  display:    flex;
  flex-direction: column;
  justify-content: flex-end;
}

section.cover::before {
  width:   100%;
  height:  5px;
  background: linear-gradient(90deg, var(--green) 0%, var(--blue) 100%);
  top: 0;
  right: auto;
  left: 0;
}

section.cover::after { color: rgba(255,255,250,.4); }

section.cover h1 {
  font-size:   4.5rem;
  color:       var(--core-white);
  max-width:   75%;
  line-height: 1.05;
}

section.cover h2 {
  font-family: var(--font-sans);
  font-size:   1.25rem;
  font-weight: 400;
  color:       var(--light-green);
  margin-top:  0.6em;
  letter-spacing: 0;
}

section.cover p {
  color:     rgba(255,255,250,.65);
  font-size: 0.9rem;
}

/* ── SECTION DIVIDER  (bright green) ────────────────────────────── */
section.section {
  background: var(--green);
  color:      var(--dark-green);
  display:    flex;
  flex-direction: column;
  justify-content: center;
  align-items: flex-start;
  padding:    80px 96px;
}

section.section::before { background: var(--dark-green); }
section.section::after  { color: rgba(10,42,31,.5); }

section.section h1,
section.section h2 {
  color:       var(--dark-green);
  font-size:   3rem;
  max-width:   70%;
}

section.section h3 {
  color:       var(--blue-grey);
  margin-bottom: 0.5em;
}

section.section p { color: rgba(10,42,31,.8); }

/* ── DARK  (dark blue-grey) ─────────────────────────────────────── */
section.dark {
  background: var(--dark-blue-grey);
  color:      var(--core-white);
}

section.dark::before { background: var(--green); }
section.dark::after  { color: rgba(255,255,250,.4); }

section.dark h1,
section.dark h2 { color: var(--core-white); }
section.dark h3 { color: var(--light-blue-grey); }
section.dark strong { color: var(--light-green); }

section.dark code {
  background: rgba(255,255,250,.10);
  color:      var(--light-green);
}

section.dark li::marker { color: var(--green); }

section.dark blockquote {
  border-left-color: var(--green);
  background:        rgba(99,255,145,.10);
  color:             rgba(255,255,250,.8);
}

/* ── LIGHT  (default, off-white) ────────────────────────────────── */
section.light {
  background: var(--offwhite-bg);
  color:      var(--ink);
}

/* ── CLOSING  (gradient, centered) ─────────────────────────────── */
section.closing {
  background: linear-gradient(135deg, var(--dark-blue-grey) 0%, var(--dark-green) 60%, var(--blue-grey) 100%);
  color:       var(--core-white);
  display:     flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align:  center;
  padding:     80px;
}

section.closing::before {
  width:  100%;
  height: 5px;
  background: linear-gradient(90deg, var(--blue) 0%, var(--green) 100%);
  top:   0;
  right: auto;
  left:  0;
}

section.closing::after { color: rgba(255,255,250,.35); }

section.closing h1 {
  font-size: 3.8rem;
  color:     var(--core-white);
}

section.closing h2 {
  color:       var(--light-green);
  font-weight: 400;
  font-size:   1.3rem;
}

section.closing p {
  color:     rgba(255,255,250,.6);
  font-size: 0.9rem;
}

/* ─── Utility Classes ──────────────────────────────────────────── */

/* Green pill badge */
.badge {
  display:        inline-block;
  background:     var(--green);
  color:          var(--dark-green);
  font-size:      0.7em;
  font-weight:    700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding:        0.25em 0.75em;
  border-radius:  999px;
  margin-bottom:  0.6em;
}

/* Stat callout */
.stat {
  font-family: var(--font-display);
  font-size:   3.5rem;
  color:       var(--dark-green);
  line-height: 1;
  display:     block;
}

.stat-label {
  font-size:   0.8rem;
  color:       var(--ink-soft);
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

/* Two-column grid (use inside a slide) */
.cols {
  display:               grid;
  grid-template-columns: 1fr 1fr;
  gap:                   var(--gap);
  align-items:           start;
}

.cols-3 {
  display:               grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap:                   var(--gap);
  align-items:           start;
}

/* Card */
.card {
  background:    var(--core-white);
  border:        1px solid rgba(43,59,100,.18);
  border-radius: var(--radius);
  padding:       1.25em 1.5em;
  box-shadow:    0 2px 12px rgba(10,42,31,.06);
}

/* Tag / label */
.tag {
  display:        inline-block;
  background:     rgba(112,153,255,.15);
  color:          var(--blue-grey);
  font-size:      0.72em;
  font-weight:    600;
  padding:        0.2em 0.65em;
  border-radius:  6px;
  letter-spacing: 0.04em;
}

/* Highlight box */
.highlight {
  background:    linear-gradient(135deg, rgba(99,255,145,.14) 0%, rgba(112,153,255,.10) 100%);
  border-left:   4px solid var(--green);
  border-radius: 0 var(--radius) var(--radius) 0;
  padding:       1em 1.5em;
}
</style>

---
<!-- _class: cover -->

<div class="logo"></div>

# The Platform<br>for AI Success

## Subtitle or Presenter Name · Event · Date

<p>dataiku.com</p>

---

<!-- Default light slide -->

## Slide Title Goes Here

Use this layout for **standard content** slides. The green accent bar at the top anchors the Dataiku brand on every slide without being distracting.

- First bullet point with key insight
- Second point supporting the narrative
- Third point — keep lists to 3–5 items maximum

> "A blockquote stands out with the green left border and gentle tinted background."

---

## Two-Column Layout

<div class="cols">
<div>

### Left Column

Use left for the **primary content** — the argument, the data, the main insight.

- Point one
- Point two
- Point three

</div>
<div>

### Right Column

Use right for supporting evidence, an image, a chart, or a complementary list.

```python
# Code example
result = model.predict(df)
print(result.head())
```

</div>
</div>

---

## Three Stats Layout

<div class="cols-3">
<div class="card">
<span class="stat">1 in 4</span>
<span class="stat-label">of Forbes Global 2000 companies trust Dataiku</span>
</div>
<div class="card">
<span class="stat">1,250+</span>
<span class="stat-label">employees across 13 global offices</span>
</div>
<div class="card">
<span class="stat">10+</span>
<span class="stat-label">years accelerating enterprise AI</span>
</div>
</div>

Use **DM Serif Display** numerals for maximum visual impact on key metrics.

---

## Cards and Badges

<span class="badge">New Feature</span>

<div class="cols">
<div class="card">

### Orchestration

Connects every tool, team, and environment in a single governed workflow.

<span class="tag">Platform</span>

</div>
<div class="card">

### Governance

Audit-ready oversight with centralised risk controls and cost visibility.

<span class="tag">Enterprise</span>

</div>
</div>

---

## Highlighted Callout

<div class="highlight">

**Key insight:** Use the highlight box to surface the single most important takeaway from a content-heavy slide. Keep it to one or two sentences maximum.

</div>

Supporting body copy follows below the callout. The gradient tint blends the brand green and blue, reinforcing the palette without overwhelming the message.

---

<!-- _class: section -->

<div class="logo"></div>

# Section Divider

### Chapter 2 — Governance

---
<!-- _class: dark -->

<div class="logo"></div>

## Dark Slide — Dark Blue Grey

Use dark slides to **add visual contrast** within a deck, or to separate major sections.

- The bright green `#63FF91` replaces the dark green accent for legibility on dark surfaces
- Strong contrast between cool off-white body text and the dark blue-grey background
- `inline code` reads well against the dark background

> Quote callouts use the green border to maintain visual rhythm on this darker canvas.

---
<!-- _class: closing -->

<div class="logo"></div>

# Thank You

## questions@dataiku.com · dataiku.com

<p>© 2026 Dataiku. All rights reserved.</p>

---

<!--
════════════════════════════════════════════════════════════════
  DATAIKU BRAND QUICK REFERENCE
════════════════════════════════════════════════════════════════

  COLOR PALETTE
  ─────────────────────────────────────────────────────────────
  Core Black       #0F1314   — body text on light surfaces
  Core White       #FFFFFA   — text on dark surfaces
  Dark Green       #0A2A1F   — primary dark background, headings
  Green            #63FF91   — primary accent, badges, highlights
  Light Green      #D1FFE6   — light accents on dark slides
  Blue             #7099FF   — secondary accent, gradients
  Dark Grey        #1D2220   — near-black alternative
  Dark Blue Grey   #081030   — deepest dark surface (closing slide)
  Blue Grey        #2B3B64   — muted dark accent, h3, tags
  Light Blue Grey  #ABCFFA   — dividers, muted text on dark

  TYPOGRAPHY
  ─────────────────────────────────────────────────────────────
  DM Serif Display          — h1 on cover & closing slides
  DM Sans 700               — h2 headings
  DM Sans 600               — h3 / uppercase labels
  DM Sans 400               — body copy
  DM Mono                   — code snippets

  SLIDE CLASSES                              LOGO VARIANT
  ─────────────────────────────────────────────────────────────
  (none / .light)           Default off-white   Black logo
  .cover                    Dark green hero     White logo
  .section                  Bright green        White logo
  .dark                     Dark blue-grey      White logo
  .closing                  Gradient            White (centered)

  UTILITY CLASSES
  ─────────────────────────────────────────────────────────────
  .badge                    Green pill label
  .stat / .stat-label       Large display metric + caption
  .cols                     Two-column grid
  .cols-3                   Three-column grid
  .card                     White card with border + shadow
  .tag                      Blue inline tag
  .highlight                Green-left-border callout box

  MARP CLI USAGE
  ─────────────────────────────────────────────────────────────
  marp --html --watch dataiku-marp-theme.md      # live preview
  marp --html --pdf  dataiku-marp-theme.md       # PDF export
  marp --html --pptx dataiku-marp-theme.md       # PowerPoint export

  NOTE: --html flag is required to render the <style> block and
  custom utility class divs.
════════════════════════════════════════════════════════════════
-->
