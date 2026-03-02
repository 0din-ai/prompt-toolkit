/**
 * Tests for error types.
 */

import {
  SigError,
  ConfigError,
  ProviderError,
  ModelError,
  InvalidInputError,
  parseSignatureString,
} from '../src';

describe('Error Types', () => {
  it('should have all error types inherit from SigError', () => {
    expect(new ConfigError('test')).toBeInstanceOf(SigError);
    expect(new ProviderError('test')).toBeInstanceOf(ProviderError);
    expect(new ModelError('test')).toBeInstanceOf(ModelError);
    expect(new InvalidInputError('test')).toBeInstanceOf(InvalidInputError);
  });

  it('should have all error types inherit from Error', () => {
    expect(new SigError('test')).toBeInstanceOf(Error);
    expect(new ConfigError('test')).toBeInstanceOf(Error);
    expect(new ProviderError('test')).toBeInstanceOf(Error);
    expect(new ModelError('test')).toBeInstanceOf(Error);
    expect(new InvalidInputError('test')).toBeInstanceOf(Error);
  });

  it('should allow ConfigError to be thrown and caught', () => {
    expect(() => {
      throw new ConfigError('Invalid configuration');
    }).toThrow(ConfigError);

    try {
      throw new ConfigError('Invalid configuration');
    } catch (error) {
      expect(error).toBeInstanceOf(ConfigError);
      expect((error as ConfigError).message).toBe('Invalid configuration');
    }
  });

  it('should allow ProviderError to be thrown and caught', () => {
    expect(() => {
      throw new ProviderError('API failure');
    }).toThrow(ProviderError);

    try {
      throw new ProviderError('API failure');
    } catch (error) {
      expect(error).toBeInstanceOf(ProviderError);
      expect((error as ProviderError).message).toBe('API failure');
    }
  });

  it('should allow ModelError to be thrown and caught', () => {
    expect(() => {
      throw new ModelError('Model not found');
    }).toThrow(ModelError);

    try {
      throw new ModelError('Model not found');
    } catch (error) {
      expect(error).toBeInstanceOf(ModelError);
      expect((error as ModelError).message).toBe('Model not found');
    }
  });

  it('should allow InvalidInputError to be thrown and caught', () => {
    expect(() => {
      throw new InvalidInputError('Invalid input');
    }).toThrow(InvalidInputError);

    try {
      throw new InvalidInputError('Invalid input');
    } catch (error) {
      expect(error).toBeInstanceOf(InvalidInputError);
      expect((error as InvalidInputError).message).toBe('Invalid input');
    }
  });

  it('should allow catching SigError to catch all subtypes', () => {
    const errors = [
      new ConfigError('test'),
      new ProviderError('test'),
      new ModelError('test'),
      new InvalidInputError('test'),
    ];

    for (const error of errors) {
      try {
        throw error;
      } catch (e) {
        expect(e).toBeInstanceOf(SigError);
      }
    }
  });

  it('should throw InvalidInputError for invalid signature prefix', () => {
    expect(() => {
      parseSignatureString('invalid');
    }).toThrow(InvalidInputError);

    try {
      parseSignatureString('invalid');
    } catch (error) {
      expect(error).toBeInstanceOf(InvalidInputError);
      expect((error as InvalidInputError).message).toContain('Invalid signature prefix');
    }
  });

  it('should throw InvalidInputError for unsupported version', () => {
    expect(() => {
      parseSignatureString('0din-v99:abcd1234');
    }).toThrow(InvalidInputError);

    try {
      parseSignatureString('0din-v99:abcd1234');
    } catch (error) {
      expect(error).toBeInstanceOf(InvalidInputError);
      expect((error as InvalidInputError).message).toContain('Unsupported signature version');
    }
  });

  it('should throw InvalidInputError for invalid hex signature', () => {
    expect(() => {
      parseSignatureString('0din-v1:notahex!');
    }).toThrow(InvalidInputError);

    try {
      parseSignatureString('0din-v1:notahex!');
    } catch (error) {
      expect(error).toBeInstanceOf(InvalidInputError);
      expect((error as InvalidInputError).message).toContain('Invalid hex signature');
    }
  });

  it('should have correct error names', () => {
    expect(new SigError('test').name).toBe('SigError');
    expect(new ConfigError('test').name).toBe('ConfigError');
    expect(new ProviderError('test').name).toBe('ProviderError');
    expect(new ModelError('test').name).toBe('ModelError');
    expect(new InvalidInputError('test').name).toBe('InvalidInputError');
  });
});
