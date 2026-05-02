import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [
    react({
      // Ensure JSX is handled in both .jsx and .js files
      include: /\.(jsx|js)$/
    })
  ],
  esbuild: {
    jsx: "automatic"
  },
  server: {
    port: 5173
  }
});

