#!/usr/bin/env node
import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';

const PROJECT_DIR = '/sessions/friendly-festive-bardeen/shorts/emu-war';
const WEB_DIR = path.join(PROJECT_DIR, 'web');
const FRAMES_DIR = path.join(PROJECT_DIR, 'frames');
const STILLS_DIR = path.join(PROJECT_DIR, 'stills');

function parseRange(arg) {
  if (arg.includes('-')) {
    const [start, end] = arg.split('-').map(Number);
    return Array.from({ length: end - start + 1 }, (_, i) => start + i);
  }
  return arg.split(',').map(Number);
}

async function renderFrames(frames, mode = 'frames') {
  const outDir = mode === 'stills' ? STILLS_DIR : FRAMES_DIR;
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

  const browser = await chromium.launch({
    headless: true,
    args: [
      '--use-angle=swiftshader',
      '--enable-unsafe-swiftshader',
      '--ignore-gpu-blocklist',
      '--disable-gpu-sandbox',
      '--no-sandbox'
    ]
  });

  const page = await browser.newPage({
    viewport: { width: 1080, height: 1920, deviceScaleFactor: 1 }
  });

  page.on('console', msg => {
    if (msg.type() === 'log') console.log('[page]', msg.text());
  });
  page.on('pageerror', err => console.error('[page error]', err));

  const fileUrl = 'http://localhost:8080/index.html';
  await page.goto(fileUrl, { waitUntil: 'networkidle', timeout: 60000 });

  await page.waitForFunction(() => window.ready === true, { timeout: 60000 });
  console.log('Renderer ready, starting render...');

  const startTime = Date.now();
  let rendered = 0;

  for (const f of frames) {
    const outFile = path.join(outDir, `f${String(f).padStart(4, '0')}.jpg`);

    if (process.env.RESUME && fs.existsSync(outFile)) {
      console.log(`[${rendered + 1}/${frames.length}] Frame ${f}: SKIP (exists)`);
      rendered++;
      continue;
    }

    try {
      await page.evaluate(frame => window.renderFrame(frame), f);
      await page.waitForTimeout(50);
      const buffer = await page.locator('#canvas').screenshot({ type: 'jpeg', quality: 95 });
      fs.writeFileSync(outFile, buffer);

      rendered++;
      const elapsed = (Date.now() - startTime) / 1000;
      const fps = rendered / elapsed;
      const eta = (frames.length - rendered) / fps;
      console.log(`[${rendered}/${frames.length}] Frame ${f}: ${fps.toFixed(1)} fps, ETA ${eta.toFixed(0)}s`);
    } catch (err) {
      console.error(`Frame ${f} failed:`, err.message);
    }
  }

  await browser.close();
  console.log(`Done: ${rendered} frames rendered in ${((Date.now() - startTime) / 1000).toFixed(1)}s`);
}

async function main() {
  const args = process.argv.slice(2);
  if (args.length < 2) {
    console.log('Usage: node render.mjs <frames|stills> "<range>" [--resume]');
    process.exit(1);
  }

  const mode = args[0];
  const range = args[1];
  const frames = parseRange(range);

  await renderFrames(frames, mode);
}

main().catch(console.error);
