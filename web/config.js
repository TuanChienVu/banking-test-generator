// Configuration for different environments
const config = {
    // Development (local)
    development: {
        API_URL: 'http://localhost:8000',
        APP_NAME: 'Banking Test Generator (Dev)',
    },
    
    // Production (deployed)
    production: {
        // Heroku backend URL (will be updated after deployment)
        API_URL: 'https://vutuanchien-testgen.herokuapp.com', // Will update after Heroku deployment
        APP_NAME: 'Banking Test Generator | Vu Tuan Chien',
    }
};

// Auto-detect environment
const isProduction = window.location.hostname !== 'localhost' 
                    && window.location.hostname !== '127.0.0.1';

// Export current config
window.APP_CONFIG = isProduction ? config.production : config.development;

console.log(`Running in ${isProduction ? 'production' : 'development'} mode`);
console.log(`API URL: ${window.APP_CONFIG.API_URL}`);
