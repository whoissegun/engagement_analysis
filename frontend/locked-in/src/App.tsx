import React from 'react';
import { motion } from 'framer-motion';
import Header from './components/Header';
import WebcamComponent from './components/WebcamComponent';
import AboutSection from './components/AboutSection';
import TeamSection from './components/TeamSection';

function App() {
  return (
    <div className="min-h-screen">
      <motion.div
        initial={{ y: '-100%' }}
        animate={{ y: 0 }}
        transition={{ duration: 1, ease: 'easeOut' }}
        className="fixed inset-0 wave-animation -z-10 opacity-20"
      />
      
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        <motion.section
          id="home"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12 scroll-mt-20"
        >
          <h1 className="text-4xl md:text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-primary to-secondary">
            Track Your Engagement
          </h1>
          <p className="text-lg text-secondary max-w-2xl mx-auto">
            Lock'd In helps you monitor and improve your focus during online sessions
            using advanced engagement tracking technology.
          </p>
        </motion.section>

        <WebcamComponent />
        <AboutSection />
        <TeamSection />
      </main>
    </div>
  );
}

export default App;