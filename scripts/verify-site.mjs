import { existsSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

const root = process.cwd();
const siteDir = join(root, "site");
const indexPath = join(siteDir, "index.html");
const cssPath = join(siteDir, "styles.css");
const jsPath = join(siteDir, "script.js");

const failures = [];

function fail(message) {
  failures.push(message);
}

function readRequired(path) {
  if (!existsSync(path)) {
    fail(`missing file: ${path}`);
    return "";
  }
  return readFileSync(path, "utf8");
}

const html = readRequired(indexPath);
const css = readRequired(cssPath);
const js = readRequired(jsPath);

if (!/<meta\s+name="viewport"[^>]+width=device-width/i.test(html)) {
  fail("index.html is missing a responsive viewport meta tag");
}

if (!/<main\s+id="main"/i.test(html)) {
  fail("index.html must expose a #main landmark for the skip link");
}

if (!/prefers-reduced-motion:\s*reduce/.test(css)) {
  fail("styles.css must honor prefers-reduced-motion");
}

if (/transition\s*:\s*all\b/i.test(css)) {
  fail("styles.css must not use transition: all");
}

if (/z-index\s*:\s*9{3,}/i.test(css)) {
  fail("styles.css must use the semantic z-index scale, not arbitrary 999 values");
}

if (/#[0-9a-f]{3,8}\b/i.test(css)) {
  fail("styles.css should use OKLCH tokens instead of raw hex colors");
}

if (!/@media\s*\(hover:\s*hover\)\s*and\s*\(pointer:\s*fine\)/.test(css)) {
  fail("hover motion must be gated behind hover and fine pointer media queries");
}

if (!/aria-pressed="true"/.test(html) || !/aria-pressed="false"/.test(html)) {
  fail("artifact tabs must expose aria-pressed state");
}

const imageMatches = [...html.matchAll(/<img\b[^>]*src="([^"]+)"[^>]*>/gi)];
if (imageMatches.length === 0) {
  fail("site must include meaningful image assets");
}

for (const match of imageMatches) {
  const [tag, src] = match;
  const alt = tag.match(/\salt="([^"]+)"/i)?.[1] ?? "";
  if (!alt.trim()) {
    fail(`image ${src} is missing descriptive alt text`);
  }
  const assetPath = join(siteDir, src);
  if (!existsSync(assetPath)) {
    fail(`referenced image does not exist: ${src}`);
    continue;
  }
  if (statSync(assetPath).size === 0) {
    fail(`referenced image is empty: ${src}`);
  }
}

for (const ref of ["styles.css", "script.js"]) {
  if (!html.includes(ref)) {
    fail(`index.html does not reference ${ref}`);
  }
}

if (!/navigator\.clipboard\.writeText/.test(js)) {
  fail("script.js should keep copy buttons functional");
}

if (failures.length > 0) {
  console.error("Site verification failed:");
  for (const failure of failures) {
    console.error(`- ${failure}`);
  }
  process.exit(1);
}

console.log("Site verification passed.");
