import React from 'react';
import { motion } from 'framer-motion';
import { Focus, Brain, Zap } from 'lucide-react';

const features = [
  {
    icon: Focus,
    title: 'Real-time Tracking',
    description: 'Monitor your engagement levels in real-time with our advanced webcam integration.'
  },
  {
    icon: Brain,
    title: 'AI-Powered Analysis',
    description: 'Our sophisticated AI algorithms analyze your facial expressions and body language.'
  },
  {
    icon: Zap,
    title: 'Instant Feedback',
    description: 'Receive immediate feedback and suggestions to improve your focus and engagement.'
  }
];

const AboutSection = () => {
  return (
    <motion.section
      id="about"
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true }}
      className="py-16 scroll-mt-20"
    >
      <div className="max-w-4xl mx-auto text-center">
        <h2 className="text-3xl md:text-4xl font-bold mb-8 bg-clip-text text-transparent bg-gradient-to-r from-primary to-secondary">
          About Lock'd In
        </h2>
        <p className="text-lg text-secondary mb-12 max-w-2xl mx-auto">
          Lock'd In revolutionizes the way you stay focused during online sessions.
          Our cutting-edge technology helps you maintain peak engagement levels.
        </p>

        <div className="grid md:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.2 }}
              className="p-6 rounded-xl bg-card-bg backdrop-blur-sm border border-card-border"
            >
              <feature.icon className="w-12 h-12 mx-auto mb-4 text-primary" />
              <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
              <p className="text-secondary">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </motion.section>
  );
};

export default AboutSection;