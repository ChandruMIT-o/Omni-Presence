import * as React from "react";
import PropTypes from "prop-types";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import TokenSelector from "./tokenSection";
import ModelList from "./ModelList";
import ModelView from "./ModelView";

function DashNav({ pathname }) {
	return (
		<Box
			sx={{
				py: 4,
				display: "flex",
				flexDirection: "column",
				alignItems: "center",
				textAlign: "center",
			}}
		>
			{pathname === "/market" && (
				<Typography variant="h5">Live Market Data</Typography>
			)}
			{pathname === "/tokens" && <TokenSelector />}
			{pathname === "/modellist" && <ModelList />}
			{pathname === "/modelview" && <ModelView />}
		</Box>
	);
}

DashNav.propTypes = {
	pathname: PropTypes.string.isRequired,
};

export default DashNav;
