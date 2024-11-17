import React, { useState, useEffect } from "react";
import { Menu, Sun, Moon } from "lucide-react";
import { motion } from "framer-motion";

const Header = () => {
  const [isDark, setIsDark] = useState(true);
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  useEffect(() => {
    document.documentElement.setAttribute(
      "data-theme",
      isDark ? "dark" : "light"
    );
  }, [isDark]);

  const toggleTheme = () => setIsDark(!isDark);
  const toggleMenu = () => setIsMenuOpen(!isMenuOpen);

  return (
    <header className="sticky top-0 z-50 py-6 px-4 md:px-8 backdrop-blur-md bg-gradient-to-r from-bg-gradient-from/80 to-bg-gradient-to/80">
      <nav className="max-w-7xl mx-auto flex items-center">
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={toggleTheme}
          className="p-2 rounded-full hover:bg-primary/10 transition-colors"
          aria-label="Toggle theme"
        >
          {isDark ? <Sun className="w-6 h-6" /> : <Moon className="w-6 h-6" />}
        </motion.button>

        <motion.h1
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          className="flex-1 text-2xl md:text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-secondary text-center"
        >
          Lock'd In
        </motion.h1>

        <div className="hidden md:flex space-x-8">
          {["Home", "About", "Team"].map((item) => (
            <motion.a
              key={item}
              href={`#${item.toLowerCase()}`}
              whileHover={{ scale: 1.1 }}
              className="text-secondary hover:text-primary transition-colors"
            >
              {item}
            </motion.a>
          ))}
        </div>

        <button
          onClick={toggleMenu}
          className="md:hidden p-2 rounded-full hover:bg-primary/10 transition-colors"
          aria-label="Toggle menu"
        >
          <Menu className="w-6 h-6" />
        </button>
      </nav>

      {isMenuOpen && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="absolute top-full left-0 right-0 bg-bg-gradient-from/95 py-4 px-4 md:hidden backdrop-blur-md"
        >
          {["Home", "About", "Team"].map((item) => (
            <a
              key={item}
              href={`#${item.toLowerCase()}`}
              className="block py-2 text-secondary hover:text-primary transition-colors"
              onClick={() => setIsMenuOpen(false)}
            >
              {item}
            </a>
          ))}
        </motion.div>
      )}
    </header>
  );
};

export default Header;
