/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#ecfdf5',
          100: '#d1fae5',
          200: '#a7f3d0',
          300: '#6ee7b7',
          400: '#34d399',
          500: '#10b981',
          600: '#059669',
          700: '#047857',
          800: '#065f46',
          900: '#064e3b',
        },
        energy: {
          solar: '#f59e0b',
          wind: '#06b6d4',
          demand: '#ef4444',
          grid: '#8b5cf6',
          battery: '#10b981',
          export: '#3b82f6',
        }
      },
    },
  },
  plugins: [],
};
