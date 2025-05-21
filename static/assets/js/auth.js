/**
 * Authentication related functionality
 */

// --- DOM Elements ---
const loginForm = document.getElementById('login-form');
const signupForm = document.getElementById('signup-form'); // Added signup form element
const landingPage = document.getElementById('landing-page');
const chatApp = document.getElementById('chat-app');
const loginErrorMsg = document.getElementById('login-error-message'); // Assuming you add an error message div
const signupErrorMsg = document.getElementById('signup-error-message'); // Assuming you add an error message div
const signupSuccessMsg = document.getElementById('signup-success-message'); // Assuming you add a success message div

// Added elements for switching between forms (assuming you add links/buttons in your HTML)
const showSignupLink = document.getElementById('show-signup-link'); // Example: "Don't have an account?"
const showLoginLink = document.getElementById('show-login-link'); // Example: "Already have an account?"


// --- Backend API Endpoints ---
const API_LOGIN_URL = '/login';
const API_REGISTER_URL = '/register';
const API_LOGOUT_URL = '/logout';
const API_STATUS_URL = '/status';


// --- Event Listeners ---

// Listen for login form submission
if (loginForm) {
    loginForm.addEventListener('submit', handleLogin);
}

// Listen for signup form submission
if (signupForm) {
    signupForm.addEventListener('submit', handleSignup);
}

// Listen for click to show signup form
if (showSignupLink) {
    showSignupLink.addEventListener('click', (e) => {
        e.preventDefault();
        showSignupForm();
    });
}

// Listen for click to show login form
if (showLoginLink) {
    showLoginLink.addEventListener('click', (e) => {
        e.preventDefault();
        showLoginForm();
    });
}

// --- UI Transition Functions ---
function showLoginForm() {
    if (loginForm) loginForm.style.display = 'block';
    if (signupForm) signupForm.style.display = 'none';
    // Clear any messages
    if (loginErrorMsg) loginErrorMsg.textContent = '';
    if (signupErrorMsg) signupErrorMsg.textContent = '';
    if (signupSuccessMsg) signupSuccessMsg.textContent = '';
}

function showSignupForm() {
    if (loginForm) loginForm.style.display = 'none';
    if (signupForm) signupForm.style.display = 'block';
     // Clear any messages
    if (loginErrorMsg) loginErrorMsg.textContent = '';
    if (signupErrorMsg) signupErrorMsg.textContent = '';
    if (signupSuccessMsg) signupSuccessMsg.textContent = '';
}


function showChatApp() {
    if (landingPage) landingPage.style.display = 'none';
    if (chatApp) chatApp.style.display = 'grid'; // Or 'block', depending on your CSS layout
    // After showing the chat app, initialize chat UI components
    // Assuming initializeChatUI() function exists in chat.js and sets up message area etc.
    if (typeof initializeChatUI === 'function') {
        initializeChatUI(); // Call the chat UI initializer from chat.js
    } else {
        console.error("initializeChatUI function not found. Ensure chat.js is loaded correctly.");
    }
     initUserProfile(); // Initialize user details in the sidebar
}

function showLandingPage() {
    if (landingPage) landingPage.style.display = 'flex'; // Or 'block'
    if (chatApp) chatApp.style.display = 'none';
}

// --- Authentication API Calls and Handlers ---

/**
 * Check login status on page load
 */
async function checkLoginStatus() {
    console.log("Checking login status...");
    try {
        const response = await fetch(API_STATUS_URL, {
            method: 'GET',
            headers: {
                 'Content-Type': 'application/json',
            }
        });

        if (response.ok) {
            const data = await response.json();
            if (data.is_authenticated) {
                console.log("User is authenticated. Showing chat app.");
                showChatApp();
                // Note: User details like email might need a separate API call if not returned by /status
                // For now, initUserProfile might use placeholders or need adjustment
            } else {
                console.log("User is not authenticated. Showing landing page.");
                showLandingPage();
            }
        } else {
             // Handle non-OK status, though /status should ideally always return 200
            console.error('Failed to check login status:', response.status);
             showLandingPage(); // Assume not logged in on error
        }

    } catch (error) {
        console.error('Network error checking login status:', error);
        // If network error, assume server might be down or login failed
        showLandingPage();
    }
}


/**
 * Handle login form submission
 * @param {Event} e - Form submit event
 */
async function handleLogin(e) {
    e.preventDefault(); // Prevent default form submission and page reload

    const emailInput = document.getElementById('email');
    const passwordInput = document.getElementById('password');

    if (!emailInput || !passwordInput) {
        console.error("Login form inputs not found.");
        return;
    }

    const email = emailInput.value.trim();
    const password = passwordInput.value; // Get password as is (plaintext)

    // Basic client-side validation
    if (!email || !password) {
        if (loginErrorMsg) loginErrorMsg.textContent = 'Please enter both email and password';
        return;
    }

    if (loginErrorMsg) loginErrorMsg.textContent = ''; // Clear previous errors

    console.log(`Attempting login for email: ${email}`);

    try {
        const response = await fetch(API_LOGIN_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            // ⚠️ SECURITY WARNING: Sending plaintext password in the body.
            body: JSON.stringify({ email: email, password: password }),
        });

        const data = await response.json();

        if (response.ok) { // Status 200-299
            console.log('Login successful:', data.message);
            // Store user email temporarily or fetch details if needed
            // For this simple demo, we'll just rely on the session on the backend
            // But we can store the email for the UI profile if needed:
            localStorage.setItem('userEmailForUI', email); // Temporary local storage for UI display only
            showChatApp(); // Transition to chat app
        } else { // Handle errors (e.g., 401 Unauthorized, 400 Bad Request)
            console.error('Login failed:', data.error);
            if (loginErrorMsg) loginErrorMsg.textContent = data.error || 'An error occurred during login.';
            alert('Login failed. Please check your credentials and try again.');
        }
    } catch (error) {
        console.error('Network error during login:', error);
        if (loginErrorMsg) loginErrorMsg.textContent = 'Could not connect to the server. Please try again.';
        alert('Network error. Please check your connection and try again.');
    }
}

/**
 * Handle signup form submission
 * @param {Event} e - Form submit event
 */
async function handleSignup(e) {
    e.preventDefault(); // Prevent default form submission and page reload

    const nameInput = document.getElementById('signup-name');
    const emailInput = document.getElementById('signup-email');
    const passwordInput = document.getElementById('signup-password');
    const confirmPasswordInput = document.getElementById('signup-confirm');
    const termsCheckbox = document.getElementById('terms');

    if (!nameInput || !emailInput || !passwordInput || !confirmPasswordInput || !termsCheckbox) {
        console.error("Signup form inputs not found.");
        return;
    }

    const name = nameInput.value.trim();
    const email = emailInput.value.trim();
    const password = passwordInput.value; // Get password as is (plaintext)
    const confirmPassword = confirmPasswordInput.value; // Get password as is (plaintext)
    const termsAccepted = termsCheckbox.checked;


    // Basic client-side validation
    if (!name || !email || !password || !confirmPassword) {
        if (signupErrorMsg) signupErrorMsg.textContent = 'Please fill in all fields.';
        return;
    }
    if (password !== confirmPassword) {
        if (signupErrorMsg) signupErrorMsg.textContent = 'Passwords do not match.';
        return;
    }
    if (!termsAccepted) {
        if (signupErrorMsg) signupErrorMsg.textContent = 'You must agree to the Terms of Service.';
        return;
    }

    if (signupErrorMsg) signupErrorMsg.textContent = ''; // Clear previous errors
    if (signupSuccessMsg) signupSuccessMsg.textContent = ''; // Clear previous success

    console.log(`Attempting signup for email: ${email}`);

    try {
        const response = await fetch(API_REGISTER_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            // ⚠️ SECURITY WARNING: Sending plaintext password in the body.
            // Note: Backend only uses email/password according to the provided app.py example
            body: JSON.stringify({ email: email, password: password /*, fullname: name */ }),
        });

        const data = await response.json();

        if (response.ok) { // Status 200-299 (backend returns 201 on success)
            console.log('Signup successful:', data.message);
            if (signupSuccessMsg) signupSuccessMsg.textContent = data.message || 'Registration successful!';
            // Optionally clear the form
            // signupForm.reset();
            // Optionally switch to the login form after successful signup
             showLoginForm();
        } else { // Handle errors (e.g., 409 Conflict, 400 Bad Request)
            console.error('Signup failed:', data.error);
            if (signupErrorMsg) signupErrorMsg.textContent = data.error || 'An error occurred during registration.';
        }
    } catch (error) {
        console.error('Network error during signup:', error);
        if (signupErrorMsg) signupErrorMsg.textContent = 'Could not connect to the server. Please try again.';
    }
}


/**
 * Initialize user profile in sidebar
 * Gets user email from temporary storage for display
 */
function initUserProfile() {
    const userEmail = localStorage.getItem('userEmailForUI') || 'User'; // Get email stored on login
    const userName = userEmail.split('@')[0]; // Simple name extraction

    const userAvatar = document.querySelector('.user-avatar');
    const userNameElement = document.querySelector('.user-name');
    const userStatusElement = document.querySelector('.user-status'); // Assuming this element exists

    if (userAvatar && userNameElement) {
        // Set user initials in avatar
        const initials = userName.substring(0, 2).toUpperCase(); // Just take first 2 letters
        userAvatar.textContent = initials;

        // Set user name
        userNameElement.textContent = capitalizeFirstLetter(userName);

        // Set status (this is demo, assuming everyone is 'Premium' after login)
        if (userStatusElement) {
             userStatusElement.textContent = 'Authenticated'; // Or 'Basic User' etc.
             // You'd need user data from the backend to show 'Premium' if applicable
        }

    } else {
        console.warn("User profile elements not found.");
    }
}

/**
 * Logout the user
 */
async function logoutUser() {
    console.log("Attempting logout...");
    try {
        const response = await fetch(API_LOGOUT_URL, {
            method: 'POST', // Logout typically uses POST
            headers: {
                 'Content-Type': 'application/json',
            }
        });

        if (response.ok) {
            console.log('Logout successful.');
            localStorage.removeItem('userEmailForUI'); // Clear temporary email
            showLandingPage(); // Transition back to landing page
             // Potentially clear chat history UI here if desired
            if (messagesContainer) messagesContainer.innerHTML = ''; // Assuming messagesContainer exists in chat.js
            if (chatHistory) chatHistory.innerHTML = ''; // Assuming chatHistory exists in chat.js
             // Re-populate chat history with dummy data or empty state
             if (typeof populateChatHistory === 'function') populateChatHistory(); // Call chat.js function
             // Reset chat state variables in chat.js (e.g., currentChatId = null)
        } else {
            console.error('Logout failed:', response.status);
            // Even if backend fails, clear client side and show landing page for UX
            localStorage.removeItem('userEmailForUI');
            showLandingPage();
        }

    } catch (error) {
        console.error('Network error during logout:', error);
         // Even if network error, clear client side and show landing page for UX
        localStorage.removeItem('userEmailForUI');
        showLandingPage();
    }
}


/**
 * Helper function to capitalize first letter of a string
 * @param {string} string - String to capitalize
 * @returns {string} Capitalized string
 */
function capitalizeFirstLetter(string) {
    if (!string) return '';
    return string.charAt(0).toUpperCase() + string.slice(1);
}


// --- Initialization ---
// Call checkLoginStatus when the page loads to determine which UI to show
document.addEventListener('DOMContentLoaded', checkLoginStatus);

// Ensure logout is triggered from somewhere in your UI (e.g. a button click)
// Example: Assuming you have a logout button with ID 'logout-button'
// const logoutButton = document.getElementById('logout-button');
// if (logoutButton) {
//     logoutButton.addEventListener('click', logoutUser);
// }


// --- Placeholder for Error/Success Message Divs (Add these to your index.html) ---
// <div id="login-error-message" style="color: red; margin-top: 10px;"></div>
// <div id="signup-error-message" style="color: red; margin-top: 10px;"></div>
// <div id="signup-success-message" style="color: green; margin-top: 10px;"></div>
// --- Placeholder for Form Switch Links (Add these to your index.html, e.g., in form footers) ---
// <p>Don't have an account? <a href="#" id="show-signup-link">Sign Up</a></p>
// <p>Already have an account? <a href="#" id="show-login-link">Log In</a></p>