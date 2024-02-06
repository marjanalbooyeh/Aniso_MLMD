def neighbors_distance_vector(positions, neighbor_list):
    """
    Calculate the distance vector between particles and their neighbors
    Parameters
    ----------
    positions: particle positions (B, N, 3)
    neighbor_list: list of neighbors for each particle (B, N, N_neighbors)

    Returns
    -------
    dr: distance vector between particles and their neighbors (B, N, N_neighbors, 3)
    """
    B = positions.shape[0]
    N = positions.shape[1]
    N_neighbors = neighbor_list.shape[-1]

    NN_extended = neighbor_list.unsqueeze(-1).expand((B, N, N_neighbors, 3))
    NN_positions = torch.gather(positions, dim=1, index=NN_extended)
    return dr