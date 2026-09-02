/**
 * Tests for OnnxProvider.create() — model parameter threading (0DIN-2097).
 *
 * OnnxProvider.create()'s `model` parameter used to be cosmetic: every cache
 * call inside create() hardcoded the literal version key "v1", so a
 * caller-supplied repo override never changed what got downloaded or loaded
 * for inference. These tests assert the resolved model name (a repo id like
 * "intfloat/multilingual-e5-large" or a caller override like
 * "test-org/custom-embedding-model") flows into every ModelCache call site:
 * downloadModel, getModelPath, modelDirectory, loadConfig.
 *
 * onnxruntime-node and @huggingface/transformers are not installed in the
 * test environment (see jest.config.js). onnxruntime-node is redirected to
 * a manual mock via moduleNameMapper; @huggingface/transformers has no such
 * mock and is stubbed per-test via jest.doMock({ virtual: true }), following
 * the pattern established in test/susfactor-create.test.ts.
 */

function mockOrt(): void {
  jest.doMock('onnxruntime-node', () => ({
    Tensor: class {},
    InferenceSession: {
      create: jest.fn().mockResolvedValue({ run: jest.fn() }),
    },
  }));
}

function mockHfTransformers(): void {
  jest.doMock(
    '@huggingface/transformers',
    () => ({
      AutoTokenizer: {
        from_pretrained: jest.fn().mockResolvedValue(
          (_text: string, _opts: unknown) => ({
            input_ids: { data: new BigInt64Array([1n]) },
            attention_mask: { data: new BigInt64Array([1n]) },
          }),
        ),
      },
      env: {},
    }),
    { virtual: true },
  );
}

function freshCache(): any {
  const { ModelCache: FreshModelCache } = require('../src/providers/model-cache');
  const cache = new FreshModelCache('/fake/cache');
  cache.downloadModel = jest.fn().mockResolvedValue(undefined);
  cache.getModelPath = jest.fn((v: string) => `/fake/cache/${v}/onnx/model.onnx`);
  cache.modelDirectory = jest.fn((v: string) => `/fake/cache/${v}`);
  cache.loadConfig = jest.fn().mockReturnValue({});
  return cache;
}

describe('OnnxProvider.create — model parameter threading', () => {
  afterEach(() => {
    jest.dontMock('onnxruntime-node');
    try {
      jest.dontMock('@huggingface/transformers');
    } catch {
      // not mocked in this test — nothing to undo
    }
    jest.resetModules();
  });

  it('threads a custom model id into every cache call instead of the "v1" literal', async () => {
    jest.resetModules();
    mockOrt();
    mockHfTransformers();
    const { OnnxProvider: FreshOnnxProvider } = require('../src/providers/onnx');
    const cache = freshCache();
    const customModel = 'test-org/custom-embedding-model';

    const provider = await FreshOnnxProvider.create(cache, customModel);

    expect(cache.downloadModel).toHaveBeenCalledWith(customModel);
    expect(cache.getModelPath).toHaveBeenCalledWith(customModel);
    expect(cache.modelDirectory).toHaveBeenCalledWith(customModel);
    expect(cache.loadConfig).toHaveBeenCalledWith(customModel);

    expect(cache.downloadModel).not.toHaveBeenCalledWith('v1');
    expect(cache.getModelPath).not.toHaveBeenCalledWith('v1');
    expect(cache.modelDirectory).not.toHaveBeenCalledWith('v1');
    expect(cache.loadConfig).not.toHaveBeenCalledWith('v1');

    expect(provider.model()).toBe(customModel);
  });

  it('defaults to the embedding repo id when no override is given', async () => {
    jest.resetModules();
    mockOrt();
    mockHfTransformers();
    const { OnnxProvider: FreshOnnxProvider } = require('../src/providers/onnx');
    const cache = freshCache();
    const defaultModel = 'intfloat/multilingual-e5-large';

    const provider = await FreshOnnxProvider.create(cache);

    expect(cache.downloadModel).toHaveBeenCalledWith(defaultModel);
    expect(cache.getModelPath).toHaveBeenCalledWith(defaultModel);
    expect(cache.modelDirectory).toHaveBeenCalledWith(defaultModel);
    expect(cache.loadConfig).toHaveBeenCalledWith(defaultModel);
    expect(provider.model()).toBe(defaultModel);
  });
});
