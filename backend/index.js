// Import required modules
const express = require('express');
const axios = require('axios');

// Create an Express application
const app = express();

// Define a route handler for the root path
app.get('/', (req, res) => {
  res.send('Hello, World!');
});







const options = {
  method: 'GET',
  url: 'https://irctc1.p.rapidapi.com/api/v3/trainBetweenStations',
  params: {
    fromStationCode: 'BVI',
    toStationCode: 'NDLS',
    dateOfJourney: '2024-02-23'
  },
  headers: {
    'X-RapidAPI-Key': 'c07c12fb5emsh7fd03ac0f0f9f9ep116accjsn36fbd19dc311',
    'X-RapidAPI-Host': 'irctc1.p.rapidapi.com'
  }
};

try {
	const response = await axios.request(options);
	console.log(response.data);
} catch (error) {
	console.error(error);
}








// Start the server and listen on port 3000
app.listen(3000, () => {
  console.log('Server is running on http://localhost:3000');
});
