import React, { useState } from "react";
import {
	Box,
	Table,
	TableBody,
	TableCell,
	TableContainer,
	TableHead,
	TableRow,
	Paper,
	Button,
	Dialog,
	DialogActions,
	DialogContent,
	DialogContentText,
	DialogTitle,
	Typography,
} from "@mui/material";

function ModelList() {
	// Sample models data
	const [models, setModels] = useState([
		{
			id: 1,
			modelName: "BTC",
			status: "Running",
			runningTime: "Created Just Now!",
		},
		{ id: 2, modelName: "ETC", status: "Paused", runningTime: "2h 5m" },
		{ id: 3, modelName: "Sol", status: "Running", runningTime: "30m" },
	]);

	// State for the delete confirmation dialog
	const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
	const [selectedModelId, setSelectedModelId] = useState(null);

	// Toggle pause/resume for a given model
	const handlePause = (modelId) => {
		setModels((prevModels) =>
			prevModels.map((model) =>
				model.id === modelId
					? {
							...model,
							status:
								model.status === "Running"
									? "Paused"
									: "Running",
						}
					: model
			)
		);
	};

	// Open delete confirmation dialog
	const handleDeleteClick = (modelId) => {
		setSelectedModelId(modelId);
		setDeleteDialogOpen(true);
	};

	// Confirm deletion of the selected model
	const handleDeleteConfirm = () => {
		setModels((prevModels) =>
			prevModels.filter((model) => model.id !== selectedModelId)
		);
		setDeleteDialogOpen(false);
		setSelectedModelId(null);
	};

	// Cancel deletion
	const handleDeleteCancel = () => {
		setDeleteDialogOpen(false);
		setSelectedModelId(null);
	};

	// Placeholder for view model action (to be updated later)
	const handleViewModel = (modelId) => {
		// Replace this alert with navigation logic when the view page is ready.
		alert("View model page for model ID: " + modelId);
	};

	return (
		<Box sx={{ width: "90%", mx: "auto", mt: 4 }}>
			<Typography variant="h5" gutterBottom>
				Online Models
			</Typography>
			<TableContainer component={Paper}>
				<Table>
					<TableHead>
						<TableRow>
							<TableCell>Model Name</TableCell>
							<TableCell>Status</TableCell>
							<TableCell>Running Time</TableCell>
							<TableCell align="center">Actions</TableCell>
						</TableRow>
					</TableHead>
					<TableBody>
						{models.map((model) => (
							<TableRow key={model.id}>
								<TableCell>{model.modelName}</TableCell>
								<TableCell>{model.status}</TableCell>
								<TableCell>{model.runningTime}</TableCell>
								<TableCell align="center">
									<Button
										variant="contained"
										onClick={() => handlePause(model.id)}
										sx={{ mr: 1 }}
									>
										{model.status === "Running"
											? "Pause"
											: "Resume"}
									</Button>
									<Button
										variant="outlined"
										onClick={() =>
											handleDeleteClick(model.id)
										}
										sx={{ mr: 1 }}
									>
										Delete
									</Button>
									<Button
										variant="outlined"
										onClick={() =>
											handleViewModel(model.id)
										}
									>
										View Model
									</Button>
								</TableCell>
							</TableRow>
						))}
					</TableBody>
				</Table>
			</TableContainer>

			<Dialog open={deleteDialogOpen} onClose={handleDeleteCancel}>
				<DialogTitle>Confirm Deletion</DialogTitle>
				<DialogContent>
					<DialogContentText>
						Are you sure you want to delete this model? This action
						cannot be undone.
					</DialogContentText>
				</DialogContent>
				<DialogActions>
					<Button onClick={handleDeleteCancel} color="primary">
						Cancel
					</Button>
					<Button
						onClick={handleDeleteConfirm}
						color="primary"
						autoFocus
					>
						Delete
					</Button>
				</DialogActions>
			</Dialog>
		</Box>
	);
}

export default ModelList;
