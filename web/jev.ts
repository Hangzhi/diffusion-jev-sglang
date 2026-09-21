import { experimental_evaluate as evaluate, type Experimental_EvaluationQuestion } from 'ai';
import { mkdir, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { requireGatewayKey, safeFailure } from './gateway-env.js';

type NativeQuestion = Experimental_EvaluationQuestion | {
  type: 'noul'; instructions: Experimental_EvaluationQuestion['instructions'];
  criteria?: { true?: string | null; false?: string | null };
};

export async function evaluateJev(state: Parameters<typeof evaluate>[0]['state'], questions: Record<string, NativeQuestion>) {
  const sdkQuestions: Record<string, Experimental_EvaluationQuestion> = Object.fromEntries(
    Object.entries(questions).map(([id, q]) => [id, q.type === 'noul' ? { ...q, type: 'boolean' } : q]),
  );
  const result = await evaluate({ model: 'typesafe-ai/jev', state, questions: sdkQuestions,
    maxRetries: 0, abortSignal: AbortSignal.timeout(120_000) });
  return {
    model: result.response.modelId,
    answers: Object.fromEntries(Object.entries(result.answers).map(([id, a]) =>
      [id, a.type === 'boolean' ? { type: 'noul', noul: a.probability } : a])),
    usage: result.usage,
    // Deliberately omit SDK request/response headers and bodies.
    rounding: result.rounding,
  };
}

async function main() {
  requireGatewayKey();
  const result = await evaluateJev('The support agent issued a full refund to the customer.', {
    refunded: { type: 'noul', instructions: 'Was a refund issued?' },
    team: { type: 'choice', instructions: 'Which team handles this?', criteria: { billing: 'Charges and refunds', technical: 'Bugs and outages' } },
    resolution: { type: 'score', instructions: 'Rate the resolution.', criteria: ['Unresolved', 'Partial', 'Complete'] },
  });
  console.log(JSON.stringify(result, null, 2));
  await mkdir('../reports/gateway', { recursive: true });
  await writeFile('../reports/gateway/jev-smoke.json', JSON.stringify({ passed: true, ...result }, null, 2) + '\n');
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch(error => { console.error(safeFailure(error)); process.exitCode = 1; });
}
