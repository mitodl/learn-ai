module.exports = {
  extends: [
    "eslint-config-mitodl",
    "eslint-config-mitodl/jest",
    "plugin:react/jsx-runtime",
    "plugin:styled-components-a11y/recommended",
    "plugin:import/typescript",
    "plugin:mdx/recommended",
    "prettier",
  ],
  plugins: ["testing-library", "import", "styled-components-a11y"],
  ignorePatterns: ["**/build/**"],
  settings: {
    "jsx-a11y": {
      components: {
        Button: "button",
        ButtonLink: "a",
        ActionButton: "button",
        ActionButtonLink: "a",
      },
    },
  },
  rules: {
    "@typescript-eslint/no-restricted-imports": [
      "error",
      {
        paths: [
          /**
           * No direct imports from large "barrel files". They make Jest slow.
           *
           * For more, see:
           *  - https://github.com/jestjs/jest/issues/11234
           *  - https://github.com/faker-js/faker/issues/1114#issuecomment-1169532948
           */
          {
            name: "@faker-js/faker",
            message: "Please use @faker-js/faker/locale/en instead.",
            allowTypeImports: true,
          },
        ],
      },
    ],
    // This rule is disabled in the default a11y config, but unclear why.
    // It does catch useful errors, e.g., buttons with no text or label.
    // If it proves to be flaky, we can find other ways to check for this.
    // We need both rules below. One for normal elements, one for styled
    "jsx-a11y/control-has-associated-label": ["error"],
    "styled-components-a11y/control-has-associated-label": ["error"],
    quotes: ["error", "double", { avoidEscape: true }],
    "no-restricted-syntax": [
      "error",
      /**
       * See https://eslint.org/docs/latest/rules/no-restricted-syntax
       *
       * The selectors use "ES Query", a css-like syntax for AST querying. A
       * useful tool is  https://estools.github.io/esquery/
       */
      {
        selector:
          "Property[key.name=fontWeight][value.raw=/\\d+/], TemplateElement[value.raw=/font-weight: \\d+/]",
        message:
          "Do not specify `fontWeight` manually. Prefer spreading `theme.typography.subtitle1` or similar. If you MUST use a fontWeight, refer to `fontWeights` theme object.",
      },
      {
        selector:
          "Property[key.name=fontFamily][value.raw=/Neue Haas/], TemplateElement[value.raw=/Neue Haas/]",
        message:
          "Do not specify `fontFamily` manually. Prefer spreading `theme.typography.subtitle1` or similar. If using neue-haas-grotesk-text, this is ThemeProvider's default fontFamily.",
      },
    ],
  },
  overrides: [
    {
      files: ["./**/*.test.{ts,tsx}"],
      plugins: ["testing-library"],
      extends: ["plugin:testing-library/react"],
      rules: {
        "testing-library/no-node-access": "off",
      },
    },
  ],
};
