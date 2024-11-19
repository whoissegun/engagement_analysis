import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import Header from "./components/Header";
import AboutSection from "./components/AboutSection";
import TeamSection from "./components/TeamSection";
import WebcamComponent from "./components/WebcamComponent";

function App() {
  // const [imageUrl, setImageUrl] = useState(null);
  // const [loading, setLoading] = useState(true);
  // const [error, setError] = useState(null);

  // useEffect(() => {
  //   const fetchImage = async () => {
  //     try {
  //       const response = await fetch(
  //         "https://api.giphy.com/v1/gifs/translate?api_key=ZHWzYswlnWgv1OYNQSYeWNQmBb2fy8lA&s=paris"
  //       );

  //       if (!response.ok) {
  //         throw new Error(`HTTP error! status: ${response.status}`);
  //       }

  //       const data = await response.json();
  //       // Extract the URL of the image from the response structure
  //       const imageUrl = data.data.images.original.url; // Adjust based on actual API response
  //       setImageUrl(imageUrl);
  //     } catch (err) {
  //       console.error("Error fetching the image:", err);
  //       setError(err.message);
  //     } finally {
  //       setLoading(false);
  //     }
  //   };

  //   fetchImage();
  // }, []);

  return (
    <div className="min-h-screen">
      <motion.div
        initial={{ y: "-100%" }}
        animate={{ y: 0 }}
        transition={{ duration: 1, ease: "easeOut" }}
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
            Lock'd In helps you monitor and improve your focus during online
            sessions using advanced engagement tracking technology.
          </p>
        </motion.section>

        {/* Image Section */}
        {/* <section className="mb-12 text-center">
          {loading && <p className="text-secondary">Loading image...</p>}
          {error && <p className="text-red-500">Error: {error}</p>}
          {imageUrl && (
            <motion.img
              src={imageUrl}
              alt="Fetched from API"
              className="mx-auto rounded-lg shadow-lg"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 1 }}
            />
          )}
        </section> */}
        <WebcamComponent />
        <AboutSection />
        <TeamSection />
      </main>
    </div>
  );
}

export default App;
