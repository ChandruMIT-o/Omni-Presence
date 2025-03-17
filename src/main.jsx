import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import CryptoDashboard from "./CryptoDashboard";

createRoot(document.getElementById("root")).render(
	<StrictMode>
		<CryptoDashboard />
	</StrictMode>
);
