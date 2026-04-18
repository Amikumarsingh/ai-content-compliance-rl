/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        // Primary accent — electric violet
        accent: {
          50:  '#f5f3ff',
          100: '#ede9fe',
          200: '#ddd6fe',
          300: '#c4b5fd',
          400: '#a78bfa',
          500: '#8b5cf6',
          600: '#7c3aed',
          700: '#6d28d9',
        },
        // Secondary — cyan/teal
        teal: {
          400: '#2dd4bf',
          500: '#14b8a6',
        },
        // Base surfaces — deep navy
        n: {
          900: '#080c14',
          800: '#0d1220',
          700: '#111827',
          600: '#161f30',
          500: '#1c2840',
          400: '#243050',
          300: '#2e3d5c',
          200: '#3d5070',
          100: '#6b7fa0',
          50:  '#94a3b8',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'hero-glow': 'radial-gradient(ellipse 80% 50% at 50% -20%, rgba(139,92,246,0.15), transparent)',
      },
      boxShadow: {
        card:  '0 1px 3px rgb(0 0 0 / 0.5), inset 0 1px 0 rgb(255 255 255 / 0.03)',
        glow:  '0 0 24px rgb(139 92 246 / 0.2)',
        'glow-teal': '0 0 24px rgb(20 184 166 / 0.2)',
      },
    },
  },
  plugins: [],
}
