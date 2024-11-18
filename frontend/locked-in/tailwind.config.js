/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Helvetica Neue', 'sans-serif'],
      },
      colors: {
        primary: 'var(--primary)',
        secondary: 'var(--secondary)',
        cardBg: 'var(--card-bg)',
        cardBorder: 'var(--card-border)',
        textPrimary: 'var(--text-primary)',
        textSecondary: 'var(--text-secondary)',
      },
      backgroundImage: {
        'gradient-to-b': 'linear-gradient(135deg, var(--bg-gradient-from), var(--bg-gradient-via), var(--bg-gradient-to))',
      },
      keyframes: {
        wave: {
          '0%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
          '100%': { backgroundPosition: '0% 50%' },
        },
      },
      animation: {
        wave: 'wave 6s ease-in-out infinite', /* Updated duration */
      },
      boxShadow: {
        'camera-glow': '0 0 30px #bbd2c5, 0 0 60px #536976', /* Updated glow colors */
      },
    },
  },
  safelist: [
    'wave-animation',
    'camera-glow',
  ],
  plugins: [],
};
