import React from "react";
import { motion } from "framer-motion";
import { Github, Linkedin, Twitter } from "lucide-react";

import khizar from "./images/khizar.jpeg";
import tendi from "./images/tendi.jpeg";
import kuro from "./images/kuro.jpeg";
import divine from "./images/divine.jpg";
import { div } from "framer-motion/client";

const team = [
  {
    name: "Khizar Malik",
    role: "Frontend Engineer",
    image: khizar,
    social: {
      github: "https://github.com/khizarrm",
      linkedin: "https://www.linkedin.com/in/khizar--malik/",
      twitter: "#",
    },
  },
  {
    name: "Tendi Sambaza",
    role: "CV Engineer",
    image: tendi,
    social: { github: "#", linkedin: "#", twitter: "#" },
  },
  {
    name: "Divine Jojolola",
    role: "ML Engineer",
    image: divine,
    social: { github: "#", linkedin: "#", twitter: "#" },
  },
  {
    name: "Kuro Gboun",
    role: "Backend Engineer",
    image: kuro,
    social: { github: "#", linkedin: "#", twitter: "#" },
  },
];

const TeamSection = () => {
  return (
    <motion.section
      id="team"
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true }}
      className="py-16 scroll-mt-20"
    >
      <div className="max-w-4xl mx-auto text-center">
        <h2 className="text-3xl md:text-4xl font-bold mb-8 bg-clip-text text-transparent bg-gradient-to-r from-primary to-secondary">
          Meet Our Team
        </h2>
        <p className="text-lg text-secondary mb-12 max-w-2xl mx-auto">
          We're a passionate team of experts dedicated to helping you achieve
          your peak performance.
        </p>

        <div className="grid md:grid-cols-4 gap-8">
          {team.map((member, index) => (
            <motion.div
              key={member.name}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.2 }}
              className="group"
            >
              <div className="relative p-6 rounded-xl bg-card-bg backdrop-blur-sm border border-card-border overflow-hidden transition-transform duration-300 group-hover:scale-105">
                <img
                  src={member.image}
                  alt={member.name}
                  className="w-32 h-32 rounded-full mx-auto mb-4 object-cover border-2 border-primary/50"
                />
                <h3 className="text-xl font-semibold mb-1">{member.name}</h3>
                <p className="text-primary font-medium mb-2">{member.role}</p>
                <p className="text-secondary mb-4">{member.bio}</p>

                <div className="flex justify-center space-x-4">
                  {Object.entries(member.social).map(([platform, link]) => {
                    const Icon = {
                      github: Github,
                      linkedin: Linkedin,
                      twitter: Twitter,
                    }[platform];
                    return (
                      <motion.a
                        key={platform}
                        href={link}
                        whileHover={{ scale: 1.2 }}
                        className="text-secondary hover:text-primary transition-colors"
                        aria-label={`${member.name}'s ${platform}`}
                      >
                        <Icon className="w-5 h-5" />
                      </motion.a>
                    );
                  })}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </motion.section>
  );
};

export default TeamSection;
