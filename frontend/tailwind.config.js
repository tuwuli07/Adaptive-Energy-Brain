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
          50: '#ecf3fd',
          100: '#d1e1fa',
          200: '#a7c4f3',
          300: '#6e9ce7',
          400: '#3471d3',
          500: '#1051b9',
          600: '#053d96',
          700: '#043078',
          800: '#06285f',
          900: '#06224e',
        },
        energy: {
          solar: '#f59e0b',
          wind: '#06b6d4',
          demand: '#ef4444',
          grid: '#8b5cf6',
          battery: '#1051b9',
          export: '#3b82f6',
        }
      },
    },
  },
  plugins: [],
};
