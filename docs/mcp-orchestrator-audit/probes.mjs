// Offline diagnostic probes against the selected bridge source, without importing
// its MCP server or changing its files. All credentials and events are synthetic.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import vm from 'node:vm';

const sourcePath = path.resolve(process.argv[2] || '');
if (!process.argv[2]) throw new Error('Usage: node probes.mjs <absolute-path-to-server.js>');
const bytes = await readFile(sourcePath);
const source = bytes.toString('utf8').replace(/\r\n/g, '\n');
const names = [
  'redactSensitiveText', 'retryAfterMsFromText', 'providerErrorTypeFromText',
  'providerErrorTypeFromDiagnosticLine', 'providerErrorTypeFromStructuredEvent',
  'modelEvidenceFromEvent', 'providerDiagnosticTextFromStderr',
  'inspectOpenCodeEventStream', 'isTimeoutResult', 'classifyResultError',
  'immutableReleasePluginModeError',
];
const locations = {};
const definitions = names.map((name) => {
  const start = source.indexOf(`function ${name}(`);
  const end = source.indexOf('\n}\n', start);
  if (start < 0 || end < 0) throw new Error(`Cannot locate complete function: ${name}`);
  locations[name] = source.slice(0, start).split('\n').length;
  return source.slice(start, end + 2);
});
const context = vm.createContext({
  CONFIG: { maxAssistantResponseChars: 131072, allowExternalPlugins: false },
  process: { env: {} },
});
vm.runInContext(definitions.join('\n\n'), context, { timeout: 1000 });
const inspect = (events) => context.inspectOpenCodeEventStream(
  events.map((event) => typeof event === 'string' ? event : JSON.stringify(event)).join('\n'),
);
const finalText = (text) => ({
  type: 'text', part: { type: 'text', text, time: { end: 1 }, messageID: 'synthetic-message' },
});
const modelEvent = (modelID) => ({
  type: 'message.updated',
  properties: { info: { role: 'assistant', providerID: 'fixture', modelID } },
});
const classify = (inspection) => context.classifyResultError({
  exitCode: 0,
  dryRun: false,
  assistantFinalResponseDetected: inspection.finalResponseDetected,
  openCodeApiErrorDetected: inspection.apiErrorDetected,
  providerErrorType: inspection.providerErrorType,
  invalidEventLineCount: inspection.invalidLines,
});

// Positive controls validate the probe boundary; they do not certify the bridge.
assert.equal(inspect([modelEvent('model-a'), finalText('done')]).runtimeObservedModel, 'model-a');
assert.equal(classify(inspect([])), 'agent_empty_final_response');
assert.equal(inspect([{ type: 'error', error: { name: 'APIError', statusCode: 401 } }]).apiErrorDetected, true);
const syntheticCredential = 'ghp_' + 'A'.repeat(36);
assert.equal(context.redactSensitiveText(`api_key=${syntheticCredential}`).includes(syntheticCredential), false);
assert.equal(context.retryAfterMsFromText('Retry-After: 2'), 2000);

const switched = inspect([modelEvent('model-a'), modelEvent('model-b'), finalText('done')]);
const missing = inspect([finalText('done')]);
const malformed = inspect(['{synthetic malformed event', finalText('done')]);
const multipart = inspect([finalText('first section'), finalText('second section')]);
const result = {
  generatedAt: new Date().toISOString(),
  sourcePath,
  sourceSha256: createHash('sha256').update(bytes).digest('hex'),
  sourceLocations: locations,
  positiveControlsPassed: 5,
  observations: {
    bareGithubClassicCredentialRedacted: context.redactSensitiveText(syntheticCredential) !== syntheticCredential,
    bareGithubFineGrainedCredentialRedacted: context.redactSensitiveText('github_pat_' + 'B'.repeat(82)) !== 'github_pat_' + 'B'.repeat(82),
    laterAuthoritativeModelObserved: switched.runtimeObservedModel === 'model-b',
    observedModelForMixedStream: switched.runtimeObservedModel,
    missingRuntimeModel: missing.runtimeObservedModel === '',
    missingRuntimeModelClassification: classify(missing),
    malformedLineCount: malformed.invalidLines,
    malformedStreamClassification: classify(malformed),
    multipartRetainsFirstSection: multipart.finalText.includes('first section'),
    pinnedReleaseWithExternalPluginError: context.immutableReleasePluginModeError({
      releasePinned: true, allowExternalPlugins: true,
    }),
  },
  limitation: 'Pure-function probes only. No provider, CLI transport, real secret, or runtime model switch was exercised. Null classification means no error from classifyResultError; runOpenCode adds a mismatch check only when model evidence is present.',
};
console.log(JSON.stringify(result, null, 2));
