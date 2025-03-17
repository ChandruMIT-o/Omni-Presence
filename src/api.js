const API_BASE = "http://localhost:8000"; // Adjust the URL if necessary

export async function trainModel({
	username,
	token,
	coingeckecoinid,
	num_episodes = 100,
	max_steps = 100,
}) {
	const url = `${API_BASE}/train_model?username=${username}&token=${token}&coingeckecoinid=${coingeckecoinid}&num_episodes=${num_episodes}&max_steps=${max_steps}`;
	const response = await fetch(url);
	if (!response.ok) {
		throw new Error("Training model failed");
	}
	return response.json();
}

export async function getTradingSuggestion({
	username,
	token,
	coingeckecoinid,
}) {
	const url = `${API_BASE}/trading_suggestion?username=${username}&token=${token}&coingeckecoinid=${coingeckecoinid}`;
	const response = await fetch(url);
	if (!response.ok) {
		throw new Error("Getting trading suggestion failed");
	}
	return response.json();
}

export async function retrieveMarkdown({ username, token }) {
	const url = `${API_BASE}/retrieve_markdown?username=${username}&token=${token}`;
	const response = await fetch(url);
	if (!response.ok) {
		throw new Error("Retrieving markdown suggestion failed");
	}
	return response.json();
}
