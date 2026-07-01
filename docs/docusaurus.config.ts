import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: '0DIN Prompt Toolkit',
  tagline: 'Jailbreak detection, similarity signatures, and threat intelligence for AI prompts',
  favicon: 'img/favicon.ico',

  markdown: {
    mermaid: true,
  },
  themes: ['@docusaurus/theme-mermaid'],

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://0din-ai.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/prompt-toolkit/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: '0din-ai', // Usually your GitHub org/user name.
  projectName: 'prompt-toolkit', // Usually your repo name.

  onBrokenLinks: 'throw',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css',
      type: 'text/css',
      integrity: 'sha384-nB0miv6/jRmo5UMMR1wu3Gz6NLsoTkbqJghGIsx//Rlm+ZU03BU6SQNC66uf4l5',
      crossorigin: 'anonymous',
    },
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/', // Serve docs at the root
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/0din-ai/prompt-toolkit/tree/main/docs/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: false, // Disable blog for now
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: '0DIN Prompt Toolkit',
      logo: {
        alt: '0DIN Prompt Toolkit Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          to: '/api/core-functions',
          label: 'API',
          position: 'left',
        },
        {
          href: 'https://github.com/0din-ai/prompt-toolkit',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {
              label: 'Getting Started',
              to: '/getting-started/installation',
            },
            {
              label: 'Concepts',
              to: '/concepts/lsh-overview',
            },
            {
              label: 'Guides',
              to: '/guides/jailbreak-detection',
            },
          ],
        },
        {
          title: 'Packages',
          items: [
            {
              label: 'Rust (odin-prompt-toolkit)',
              href: 'https://github.com/0din-ai/prompt-toolkit/tree/main/packages/rust',
            },
            {
              label: 'Python (odin-prompt-toolkit)',
              href: 'https://github.com/0din-ai/prompt-toolkit/tree/main/packages/python',
            },
            {
              label: 'TypeScript (@0din/prompt-toolkit)',
              href: 'https://github.com/0din-ai/prompt-toolkit/tree/main/packages/typescript',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/0din-ai/prompt-toolkit',
            },
            {
              label: 'Specification',
              to: '/reference/spec',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} 0DIN. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['rust', 'python', 'typescript', 'toml', 'json', 'bash'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
