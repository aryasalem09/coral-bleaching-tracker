import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const pagesBase = "/coral-bleaching-tracker/";

export default defineConfig({
  plugins: [react()],
  base: process.env.GITHUB_ACTIONS === "true" ? pagesBase : "/",
});
