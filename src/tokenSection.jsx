import React, { useState, useEffect, useContext, createContext } from "react";
import {
	Box,
	Typography,
	Autocomplete,
	TextField,
	LinearProgress,
	Button,
	Slider,
	MenuItem,
	Select,
} from "@mui/material";
import ReactMarkdown from "react-markdown";

// Context for managing token-related state
const TokenContext = createContext();

export function TokenProvider({ children }) {
	const [tokenData, setTokenData] = useState({
		selectedToken: null,
		portfolioRange: [1000, 50000],
		modelType: "DQN",
		progress: 0,
		status: "",
		agentOnline: false,
	});

	return (
		<TokenContext.Provider value={{ tokenData, setTokenData }}>
			{children}
		</TokenContext.Provider>
	);
}

const modelOptions = ["DQN", "PPO", "SAC", "TD3"];

function TokenSelector() {
	const { tokenData, setTokenData } = useContext(TokenContext);
	const {
		selectedToken,
		portfolioRange,
		modelType,
		progress,
		status,
		agentOnline,
	} = tokenData;

	// New state to store tokens loaded from the CSV file
	const [tokens, setTokens] = useState([]);

	// Load tokens from tokens.csv on component mount
	useEffect(() => {
		fetch("tokens.csv")
			.then((response) => response.text())
			.then((text) => {
				const lines = text.split("\n");
				// Skip header and filter out empty lines
				const tokenData = lines
					.slice(1)
					.filter((line) => line.trim() !== "")
					.map((line) => {
						const parts = line.split(",");
						return { name: parts[0].trim(), id: parts[1].trim() };
					});
				setTokens(tokenData);
			})
			.catch((error) => {
				console.error("Error loading tokens.csv:", error);
			});
	}, []);

	const trainingSteps = [
		"Collecting past market data...",
		"Collecting past news and tweets data...",
		"Initializing model parameters...",
		"Training the model...",
		"Model evaluation...",
		"Agent is now online.",
	];

	// Simulate training progress when analysis starts
	useEffect(() => {
		let interval;
		if (progress > 0 && progress < trainingSteps.length * 20) {
			interval = setInterval(() => {
				setTokenData((prev) => ({
					...prev,
					progress: prev.progress + 20,
					status: trainingSteps[Math.floor(prev.progress / 20)],
				}));
			}, 1500);
		} else if (progress >= trainingSteps.length * 20) {
			setTokenData((prev) => ({ ...prev, agentOnline: true }));
			clearInterval(interval);
		}
		return () => clearInterval(interval);
	}, [progress, setTokenData]);

	const handleTokenChange = (event, newValue) => {
		setTokenData((prev) => ({
			...prev,
			selectedToken: newValue,
			progress: 0,
			status: "",
			agentOnline: false,
		}));
	};

	const handlePortfolioChange = (event, newValue) => {
		setTokenData((prev) => ({ ...prev, portfolioRange: newValue }));
	};

	const handleModelChange = (event) => {
		setTokenData((prev) => ({ ...prev, modelType: event.target.value }));
	};

	const startAnalysis = () => {
		if (!selectedToken) return;
		setTokenData((prev) => ({
			...prev,
			progress: 20,
			status: trainingSteps[0],
			agentOnline: false,
		}));
	};

	const resetAgent = () => {
		setTokenData({
			selectedToken: null,
			portfolioRange: [1000, 50000],
			modelType: "DQN",
			progress: 0,
			status: "",
			agentOnline: false,
		});
	};

	// Create markdown content using the selected parameters
	const insightsMarkdown = `
## Investment Recommendation: BUY

**Based on the analysis of recent market trends and whale activity, our model recommends a BUY signal.**  This recommendation is derived from the confluence of several factors detailed below.  However, it's crucial to remember that this is a model-generated suggestion, and independent research and risk assessment are strongly advised before making any investment decisions.


### Market Overview

* **Average Price:**  The current market average price stands at a significant **83,160.34**. This indicates a potentially strong market position.
* **Volatility:** Market volatility is measured at **12.72**. While some volatility is expected in the market, this level requires careful consideration and monitoring for sudden price swings.

### Recent News Analysis

The recent news provides a mixed picture, highlighting both positive and negative aspects of the market:

**Negative Sentiment:**

* **Crypto Romance Scams:** A cautionary tale highlights the risks of online scams, emphasizing the need for vigilance and verification before engaging in any online financial transactions. This underscores the importance of responsible investing and due diligence.  The loss of $10,000 in a single scam exemplifies the potential for significant financial harm.

**Positive Sentiment (with caveats):**

* **Ripple Adoption:**  News of growing $XRP adoption suggests positive momentum for Ripple, potentially influencing market sentiment. However, the inclusion of unrelated hashtags like #TRUMP and the seemingly promotional nature of the tweet require careful scrutiny.  Always independently verify such claims.
* **Solana Collection Campaign:** This news piece presents an unusual solicitation for funds.  The request, while potentially legitimate, lacks transparency and should be approached with extreme caution due to the high risk of scams in the crypto space.  Do *not* send funds without thorough verification of the project's legitimacy and identity.


### Whale Activity

* **Significant Whale Transaction:** A substantial whale transaction of **200,804.274921** has been observed. Large transactions by whales can often signal significant market movements, potentially indicating confidence in the market's future.  However, the direction of the impact (positive or negative) cannot be definitively determined from this information alone.


### Model Justification for BUY Signal

The model's "BUY" recommendation is primarily influenced by the high average market price (83,160.34) and the substantial whale transaction (200,804.274921).  These indicators suggest potential positive market momentum.  However, the high volatility (12.72) and the presence of cautionary news regarding scams require investors to proceed with caution.  The model's recommendation should be viewed as one data point among many, and not a guarantee of profit.

**Disclaimer:** This analysis is for informational purposes only and does not constitute financial advice.  Investing in cryptocurrencies involves significant risk, and you could lose some or all of your investment. Conduct thorough research and consult with a qualified financial advisor before making any investment decisions.  The inclusion of specific cryptocurrencies or projects does not constitute endorsement.`;

	return (
		<Box sx={{ width: "90%", textAlign: "center", mt: 4 }}>
			<Typography variant="h5" sx={{ mb: 2 }}>
				Select a Token for Analysis
			</Typography>

			{/* Inputs arranged in a single row */}
			<Box
				sx={{
					display: "flex",
					alignItems: "center",
					justifyContent: "center",
					gap: 2,
					flexWrap: "wrap",
				}}
			>
				{/* Token Selection using tokens from CSV */}
				<Autocomplete
					options={tokens}
					getOptionLabel={(option) => option.name}
					value={selectedToken}
					onChange={handleTokenChange}
					renderInput={(params) => (
						<TextField
							{...params}
							label="Choose a Token"
							variant="outlined"
						/>
					)}
					sx={{ width: 200 }}
				/>

				{/* Portfolio Range */}
				<Typography variant="body1">Portfolio Range:</Typography>
				<Slider
					value={portfolioRange}
					onChange={handlePortfolioChange}
					valueLabelDisplay="auto"
					min={500}
					max={100000}
					step={500}
					sx={{ width: 200 }}
				/>

				{/* Model Type */}
				<Select
					value={modelType}
					onChange={handleModelChange}
					sx={{ width: 150 }}
				>
					{modelOptions.map((model) => (
						<MenuItem key={model} value={model}>
							{model}
						</MenuItem>
					))}
				</Select>

				{/* Start Analysis Button */}
				<Button
					variant="contained"
					color="primary"
					onClick={startAnalysis}
					disabled={!selectedToken}
				>
					Start Analysis
				</Button>
			</Box>

			{/* Progress Bar and Status */}
			{progress > 0 && (
				<>
					<Typography variant="body1" sx={{ mt: 3 }}>
						{status}
					</Typography>
					<LinearProgress
						variant="determinate"
						value={(progress / (trainingSteps.length * 20)) * 100}
						sx={{ mt: 2 }}
					/>
				</>
			)}

			{/* Display Model Insights using react-markdown when agent is online */}
			{agentOnline && (
				<Box
					sx={{
						mt: 4,
						textAlign: "left",
						border: "1px solid #ccc",
						p: 2,
						borderRadius: 2,
					}}
				>
					<Typography variant="h6">Model Insights</Typography>
					<Box sx={{ mt: 2 }}>
						<ReactMarkdown>{insightsMarkdown}</ReactMarkdown>
					</Box>
					<Box
						sx={{
							display: "flex",
							justifyContent: "space-around",
							mt: 3,
						}}
					>
						<Button variant="contained" color="primary">
							Save & Deploy Agent
						</Button>
						<Button
							variant="outlined"
							color="secondary"
							onClick={resetAgent}
						>
							Create Another Agent
						</Button>
					</Box>
				</Box>
			)}
		</Box>
	);
}

export default TokenSelector;
