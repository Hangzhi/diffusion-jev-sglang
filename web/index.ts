import { generateText } from 'ai';
import { mkdir, writeFile } from 'node:fs/promises';
import { requireGatewayKey, safeFailure } from './gateway-env.js';

async function main() {
  requireGatewayKey();
  const started = performance.now();
  const { text, usage } = await generateText({
    model: 'openai/gpt-5.5',
    prompt: 'Invent a new holiday and describe its traditions.',
    maxOutputTokens: 1000,
    maxRetries: 0,
    abortSignal: AbortSignal.timeout(120_000),
  });
  if (!text.trim()) throw new Error('Model returned no text');
  console.log(text);
  await mkdir('../reports/gateway', { recursive: true });
  await writeFile('../reports/gateway/holiday.json', JSON.stringify({
    model: 'openai/gpt-5.5', passed: true, text, usage,
    latency_ms: performance.now() - started, timestamp: new Date().toISOString(),
  }, null, 2) + '\n');
}
main().catch(error => { console.error(safeFailure(error)); process.exitCode = 1; });
