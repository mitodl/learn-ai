import js from "@eslint/js"
import globals from "globals"
import tseslint from "typescript-eslint"
import react from "eslint-plugin-react"
import reactHooks from "eslint-plugin-react-hooks"
import jsxA11y from "eslint-plugin-jsx-a11y"
import importPlugin from "eslint-plugin-import"
import testingLibrary from "eslint-plugin-testing-library"
import jest from "eslint-plugin-jest"
import styledA11y from "eslint-plugin-styled-components-a11y"
import * as mdx from "eslint-plugin-mdx"
import prettier from "eslint-config-prettier"

const SOURCE = ["**/*.{js,jsx,ts,tsx}"]
const TESTS = ["**/*.{test,spec}.{js,jsx,ts,tsx}"]

export default tseslint.config(
  {
    ignores: ["**/build/**", "**/.next/**", "**/out/**", "**/node_modules/**"],
  },

  js.configs.recommended,
  ...tseslint.configs.recommended,

  {
    files: SOURCE,
    ...react.configs.flat.recommended,
    languageOptions: {
      ...react.configs.flat.recommended.languageOptions,
      globals: { ...globals.browser, ...globals.node },
    },
    settings: {
      react: { version: "19" },
      "jsx-a11y": {
        components: {
          Button: "button",
          ButtonLink: "a",
          ActionButton: "button",
          ActionButtonLink: "a",
        },
      },
    },
  },
  react.configs.flat["jsx-runtime"],
  reactHooks.configs.flat.recommended,
  jsxA11y.flatConfigs.recommended,
  importPlugin.flatConfigs.typescript,

  {
    files: SOURCE,
    plugins: { "styled-components-a11y": styledA11y },
    rules: { ...styledA11y.configs.recommended.rules },
  },

  mdx.flat,

  {
    files: TESTS,
    ...testingLibrary.configs["flat/react"],
    languageOptions: { globals: { ...globals.jest } },
  },
  {
    files: TESTS,
    ...jest.configs["flat/recommended"],
    rules: {
      ...jest.configs["flat/recommended"].rules,
      "jest/expect-expect": "off",
      "jest/no-conditional-expect": "off",
      "testing-library/no-node-access": "off",
      "@typescript-eslint/no-non-null-assertion": "off",
    },
  },

  prettier,

  {
    files: SOURCE,
    rules: {
      eqeqeq: "error",
      "no-var": "error",
      "prefer-template": "error",
      "guard-for-in": "error",
      "no-throw-literal": "error",
      camelcase: ["error", { properties: "never" }],
      "react/prop-types": "off",
      "react/display-name": "off",
      "react/no-unescaped-entities": "off",
      "no-unused-vars": "off",
      "@typescript-eslint/no-unused-vars": [
        "error",
        {
          varsIgnorePattern: "^_",
          argsIgnorePattern: "^_",
          destructuredArrayIgnorePattern: "^_",
          ignoreRestSiblings: true,
        },
      ],
      "@typescript-eslint/no-restricted-imports": [
        "error",
        {
          paths: [
            {
              name: "@faker-js/faker",
              message: "Please use @faker-js/faker/locale/en instead.",
              allowTypeImports: true,
            },
          ],
        },
      ],
      "jsx-a11y/control-has-associated-label": ["error"],
      "styled-components-a11y/control-has-associated-label": ["error"],
      quotes: ["error", "double", { avoidEscape: true }],
      "no-restricted-syntax": [
        "error",
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
  },
)
