import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/installation',
        'getting-started/quick-start',
        'getting-started/configuration',
      ],
    },
    {
      type: 'category',
      label: 'Concepts',
      items: [
        'concepts/lsh-overview',
        'concepts/signature-versions',
        'concepts/embedding-providers',
        'concepts/cm-lsh',
        'concepts/cross-language',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api/core-functions',
        'api/types',
        'api/providers',
        'api/signature-format',
        'api/errors',
        'api/cm-lsh-api',
      ],
    },
    {
      type: 'category',
      label: 'Guides',
      items: [
        'guides/duplicate-detection',
        'guides/similarity-search',
        'guides/native-acceleration',
        'guides/performance',
        'guides/migration',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'reference/spec',
        'reference/versioning',
      ],
    },
  ],
};

export default sidebars;
