import { RouterProvider } from "@tanstack/react-router";
import { Analytics } from "@vercel/analytics/react";
import { getRouter } from "./router";

const router = getRouter();

export function App() {
  return (
    <>
      <RouterProvider router={router} />
      <Analytics />
    </>
  );
}
