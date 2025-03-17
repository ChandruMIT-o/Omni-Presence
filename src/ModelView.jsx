import React, { useState } from "react";
import {
	Box,
	Typography,
	Accordion,
	AccordionSummary,
	AccordionDetails,
	Button,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ReactMarkdown from "react-markdown";

function ModelView() {
	// Initial sample insights as markdown strings
	const [insights, setInsights] = useState([
		`## Investment Recommendation: BUY

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

**Disclaimer:** This analysis is for informational purposes only and does not constitute financial advice.  Investing in cryptocurrencies involves significant risk, and you could lose some or all of your investment. Conduct thorough research and consult with a qualified financial advisor before making any investment decisions.  The inclusion of specific cryptocurrencies or projects does not constitute endorsement.`,
		"### Insight 2\n- **Metric C:** Value\n- **Metric D:** Value\n",
	]);

	// Function to simulate retrieving another insight
	const addInsight = () => {
		const newIndex = insights.length + 1;
		const newInsight = `### Insight ${newIndex}\n- **Metric X:** Value\n- **Metric Y:** Value\n`;
		setInsights((prevInsights) => [...prevInsights, newInsight]);
	};

	return (
		<Box sx={{ width: "90%", mx: "auto", mt: 4 }}>
			<Typography variant="h5" gutterBottom>
				Model Insights
			</Typography>
			{insights.map((insight, index) => (
				<Accordion key={index}>
					<AccordionSummary expandIcon={<ExpandMoreIcon />}>
						<Typography>Insight {index + 1}</Typography>
					</AccordionSummary>
					<AccordionDetails>
						<ReactMarkdown>{insight}</ReactMarkdown>
					</AccordionDetails>
				</Accordion>
			))}
			<Box sx={{ mt: 3, textAlign: "center" }}>
				<Button
					variant="contained"
					color="primary"
					onClick={addInsight}
				>
					Retrieve Another Insight
				</Button>
			</Box>
		</Box>
	);
}

export default ModelView;
