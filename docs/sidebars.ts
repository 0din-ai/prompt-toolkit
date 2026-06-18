import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

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
        'concepts/susfactor',
        'concepts/defense-in-depth',
        'concepts/signature-versions',
        'concepts/embedding-providers',
      ],
    },
    {
      type: 'category',
      label: 'Guides',
      items: [
        'guides/jailbreak-detection',
        'guides/threatfeed',
        'guides/duplicate-detection',
        'guides/similarity-search',
        'guides/native-acceleration',
        'guides/performance',
        'guides/migration',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api/core-functions',
        'api/susfactor-api',
        'api/types',
        'api/providers',
        'api/signature-format',
        'api/errors',
        'api/cm-lsh-api',
      ],
    },
    {
      type: 'category',
      label: 'Advanced',
      items: [
        'concepts/cm-lsh',
        'concepts/cross-language',
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
