import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

export default defineConfig({
  plugins: [
    react({
      exclude: [
        /[/\\]src[/\\]routes[/\\]/,
        /[/\\]src[/\\]features[/\\]screening[/\\]screening\.tsx(?:\?|$)/,
      ],
    }),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      "@": path.resolve(import.meta.dirname, "./src"),
    },
  },
  build: {
    chunkSizeWarningLimit: 1500, // Increases the warning limit to 1.5MB to account for PDF.js
  },
});
