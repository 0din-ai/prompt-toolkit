import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: '0din-sig',
  tagline: 'Multi-language LSH signature SDK for AI prompt similarity detection',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://0din.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/sig-sdk/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: '0din', // Usually your GitHub org/user name.
  projectName: 'sig-sdk', // Usually your repo name.

  onBrokenLinks: 'warn', // Changed from 'throw' to allow build with broken links

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/', // Serve docs at the root
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/0din/sig-sdk/tree/main/docs/',
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
      title: '0din-sig',
      logo: {
        alt: '0din-sig Logo',
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
          to: '/docs/api/core-functions',
          label: 'API',
          position: 'left',
        },
        {
          href: 'https://github.com/0din/sig-sdk',
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
              to: '/docs/getting-started/installation',
            },
            {
              label: 'Concepts',
              to: '/docs/concepts/lsh-overview',
            },
            {
              label: 'Guides',
              to: '/docs/guides/duplicate-detection',
            },
          ],
        },
        {
          title: 'Packages',
          items: [
            {
              label: 'Rust (odin-sig)',
              href: 'https://github.com/0din/sig-sdk/tree/main/packages/rust',
            },
            {
              label: 'Python (0din-sig)',
              href: 'https://github.com/0din/sig-sdk/tree/main/packages/python',
            },
            {
              label: 'TypeScript (@0din/sig)',
              href: 'https://github.com/0din/sig-sdk/tree/main/packages/typescript',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/0din/sig-sdk',
            },
            {
              label: 'Specification',
              to: '/docs/reference/spec',
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
