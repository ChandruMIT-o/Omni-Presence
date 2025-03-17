import * as React from "react";
import PropTypes from "prop-types";
import Box from "@mui/material/Box";
import { createTheme } from "@mui/material/styles";
import DashboardIcon from "@mui/icons-material/Dashboard";
import ShowChartIcon from "@mui/icons-material/ShowChart";
import TableChartIcon from "@mui/icons-material/TableChart";
import AnalyticsIcon from "@mui/icons-material/Analytics";
import { AppProvider } from "@toolpad/core/AppProvider";
import { DashboardLayout } from "@toolpad/core/DashboardLayout";
import { useDemoRouter } from "@toolpad/core/internal";
import DashNav from "./DashNav";
import { TokenProvider } from "./tokenSection";

const NAVIGATION = [
	{ segment: "market", title: "Market Overview", icon: <DashboardIcon /> },
	{ segment: "tokens", title: "Token Analysis", icon: <ShowChartIcon /> },
	{ segment: "modellist", title: "Model List", icon: <TableChartIcon /> },
	{ segment: "modelview", title: "Model View", icon: <AnalyticsIcon /> },
];

const cryptoTheme = createTheme({
	cssVariables: { colorSchemeSelector: "data-toolpad-color-scheme" },
	colorSchemes: { light: true, dark: true },
	breakpoints: {
		values: { xs: 0, sm: 600, md: 600, lg: 1200, xl: 1536 },
	},
});

function CryptoDashboard(props) {
	const { window } = props;
	const router = useDemoRouter("/market");
	const demoWindow = window !== undefined ? window() : undefined;

	return (
		<AppProvider
			navigation={NAVIGATION}
			branding={{
				logo: (
					<img
						src="src/assets/logo.png"
						alt="CryptoAI Logo"
						style={{ height: 35 }}
					/>
				),
				title: "Omni Presence",
				homeUrl: "/market",
			}}
			router={router}
			theme={cryptoTheme}
			window={demoWindow}
		>
			<TokenProvider>
				<DashboardLayout>
					<DashNav pathname={router.pathname} />
				</DashboardLayout>
			</TokenProvider>
		</AppProvider>
	);
}

CryptoDashboard.propTypes = {
	window: PropTypes.func,
};

export default CryptoDashboard;
