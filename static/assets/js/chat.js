/**
 * Chat functionality
 */

// DOM Elements
const messagesContainer = document.getElementById('messages-container');
const chatInputContainer = document.getElementById('chat-input-container');
const messageInput = document.getElementById('message-input');
const sendMessageBtn = document.getElementById('send-message');
const startNewChatBtn = document.getElementById('start-new-chat');
const newChatBtn = document.querySelector('.new-chat-btn'); // 'New Conversation' button in sidebar
const typingIndicator = document.getElementById('typing-indicator');
const welcomeScreen = document.getElementById('welcome-screen');
const chatHistory = document.getElementById('chat-history');
const webSearchBtn = document.getElementById('web-search-btn');

// --- Backend API Endpoint ---
// ✅ CORRECTED: Matches the /chat endpoint in your Flask app
const API_ASK_URL = '/chat';

// --- NOTE: Sample data and functions below are for demo history ONLY ---
// --- The actual message sending logic will now call the backend ---

// Sample data for chat history (STILL DEMO, NOT CONNECTED TO BACKEND PERSISTENCE)
// This data is only used to populate the sidebar list visually.
const sampleChats = [
    // { id: 1, title: "Property law questions", timestamp: "2 hours ago", preview: "What are my rights as a tenant?" },
    // { id: 2, title: "Contract review assistance", timestamp: "Yesterday", preview: "Can you help me understand this NDA?" },
    // { id: 3, title: "Employment law advice", timestamp: "3 days ago", preview: "What are the legal working hours?" },
    // { id: 4, title: "Copyright infringement", timestamp: "1 week ago", preview: "Someone is using my work without permission" },
    // { id: 5, title: "Starting a business", timestamp: "2 weeks ago", preview: "What legal structure should I choose?" }
];

// Current chat state (used for UI, not for backend history management in this version)
let currentChatId = null; // Null indicates a new conversation not yet saved/tracked by ID
// NOTE: Web search functionality is still a demo feature in the frontend
let isWebSearchActive = false;
// Keep track of uploaded file for demo UI (not sent to backend RAG)
let uploadedFile = null;


/**
 * Populate chat history with sample data
 * NOTE: This is still sample data. For persistent history, you'd need a backend
 * to save/load chat sessions.
 */
function populateChatHistory() {
    // Clear existing history
    if (chatHistory) {
        chatHistory.innerHTML = '';

         // Add event listener to the 'New Conversation' button if it exists and listener isn't already added
        // Use a flag to prevent adding multiple listeners on repeated calls
        if (newChatBtn && !newChatBtn.__listenerAdded) {
            newChatBtn.addEventListener('click', startNewChat);
            newChatBtn.__listenerAdded = true;
        }
    }


    sampleChats.forEach(chat => {
        const chatItem = document.createElement('div');
        chatItem.className = 'chat-item';
        chatItem.dataset.id = chat.id;
        chatItem.innerHTML = `
            <div class="chat-icon">
                <i class="fas fa-comment"></i>
            </div>
            <div class="chat-info">
                <div class="chat-title">${escapeHTML(shortenText(chat.title, 25))}</div>
                <div class="chat-timestamp">${escapeHTML(chat.timestamp)}</div>
            </div>
        `;

        // NOTE: Disabling click handler for demo history items
        // The backend integration currently only supports a new conversation flow.
        // To enable loading old chats, you'd need backend endpoints to fetch messages for a given chat ID.
        // chatItem.addEventListener('click', () => loadChat(chat.id));

        if (chatHistory) {
             chatHistory.appendChild(chatItem);
        }
    });

     // Add click listener to startNewChatBtn (welcome screen button) if it exists and listener isn't added
    if (startNewChatBtn && !startNewChatBtn.__listenerAdded) {
         startNewChatBtn.addEventListener('click', startNewChat);
         startNewChatBtn.__listenerAdded = true;
    }
}


/**
 * Load a chat from history (STILL DEMO)
 * This function is currently linked to sample data only.
 * A real implementation would fetch messages from the backend for the given chatId.
 * @param {number} chatId - ID of the chat to load
 */
function loadChat(chatId) {
    console.log(`Loading demo chat: ${chatId}. Note: Backend integration only supports new chats currently.`);
    // Remember current chat ID (for UI styling)
    currentChatId = chatId;

    // Remove active class from all chat items
    document.querySelectorAll('.chat-item').forEach(item => {
        item.classList.remove('active');
    });

    // Add active class to selected chat
    const selectedChat = document.querySelector(`.chat-item[data-id="${chatId}"]`);
    if (selectedChat) {
        selectedChat.classList.add('active');
    }

    // Update chat title
    const chat = sampleChats.find(c => c.id === chatId);
    if (chat) {
        // Ensure chat title element exists before updating
        const chatTitleElement = document.querySelector('.chat-title-text');
        if(chatTitleElement) {
            chatTitleElement.textContent = escapeHTML(chat.title);
        }
    }

    // Hide welcome screen, show messages and input
    if (welcomeScreen) welcomeScreen.style.display = 'none';
    if (messagesContainer) messagesContainer.style.display = 'flex';
    if (chatInputContainer) chatInputContainer.style.display = 'block';

    // Clear messages container for the "loaded" chat
    if (messagesContainer) messagesContainer.innerHTML = '';

    // --- DEMO CONTENT ONLY ---
    // In a real app, you would fetch messages for chatId from your backend API
    // For demonstration, we'll just put a placeholder
     const userMessage = document.createElement('div');
        userMessage.className = 'message outgoing';
        userMessage.innerHTML = `
            <div class="message-avatar user-message-avatar">
                <i class="fas fa-user"></i>
            </div>
            <div class="message-content">
                <div class="message-text">Loading demo chat ${chatId}... (Backend history not integrated yet)</div>
                <div class="message-info">
                    <span>${getCurrentTime()}</span>
                </div>
            </div>
        `;
        if (messagesContainer) messagesContainer.appendChild(userMessage);

    // Scroll to bottom
    scrollToBottom();

    // Hide sidebar on mobile after selection
    if (window.innerWidth <= 768) {
        // Assuming 'sidebar' is defined elsewhere (e.g., ui.js)
         const sidebar = document.getElementById('sidebar');
         if(sidebar) {
             sidebar.classList.remove('active');
         }
    }
}


/**
 * Start a new chat
 */
function startNewChat() {
    console.log("Starting new chat (clearing UI)");
    // Set current chat ID to null for new chat
    currentChatId = null; // Indicates a new unsaved conversation

    // Remove active class from all chat items
    document.querySelectorAll('.chat-item').forEach(item => {
        item.classList.remove('active');
    });

    // Update chat title
     const chatTitleElement = document.querySelector('.chat-title-text');
    if(chatTitleElement) {
        chatTitleElement.textContent = 'New Conversation';
    }


    // Hide welcome screen, show messages and input
    if (welcomeScreen) welcomeScreen.style.display = 'none';
    if (messagesContainer) messagesContainer.style.display = 'flex';
    if (chatInputContainer) chatInputContainer.style.display = 'block';

    // Clear messages container
    if (messagesContainer) messagesContainer.innerHTML = '';

    // Add welcome message
    const welcomeMessage = document.createElement('div');
    welcomeMessage.className = 'message'; // No 'outgoing' for bot/system messages
    welcomeMessage.innerHTML = `
        <div class="message-avatar bot-avatar">
            <i class="fas fa-scale-balanced"></i>
        </div>
        <div class="message-content">
            <div class="message-text">
                <p>Hello! I'm your AI-powered legal assistant. How can I help you today?</p>
                <p>You can ask me about:</p>
                <ul style="margin-top: 10px; margin-left: 20px;">
                    <li>Understanding legal procedures</li>
                    <li>Finding relevant laws and provisions</li>
                    <li>Guidance on legal rights and obligations</li>
                    <li>Explanation of legal terminology</li>
                </ul>
                <p style="margin-top: 10px; font-size: 0.9em; color: #666;">Please note: I provide AI-generated information and should not be considered legal advice. Consult with a qualified legal professional for advice specific to your situation.</p>
            </div>
            <div class="message-info">
                <span>Just now</span>
            </div>
        </div>
    `;
    if (messagesContainer) messagesContainer.appendChild(welcomeMessage);


    // Focus on input if it exists
    if (messageInput) {
       messageInput.focus();
    }

    // Hide sidebar on mobile after selection
    if (window.innerWidth <= 768) {
        // Assuming 'sidebar' is defined elsewhere (e.g., ui.js)
         const sidebar = document.getElementById('sidebar');
         if(sidebar) {
             sidebar.classList.remove('active');
         }
    }
}

/**
 * Send a message (INTEGRATED WITH FLASK BACKEND)
 */
// Add a new API endpoint constant
const API_WEB_SEARCH_URL = '/web_search'; // Add this near your existing API_ASK_URL constant

// Update the sendMessage function
async function sendMessage() {
    const message = messageInput ? messageInput.value.trim() : '';

    if (message === '') {
        if (!uploadedFile) {
            console.log("Empty message and no file.");
            return;
        }
        return;
    }

    // Create and append user message element
    const userMessageElement = document.createElement('div');
    userMessageElement.className = 'message outgoing';
    let userMessageHTML = `
        <div class="message-avatar user-message-avatar">
            <i class="fas fa-user"></i>
        </div>
        <div class="message-content">
            <div class="message-text">
                ${isWebSearchActive ? '<i class="fas fa-globe"></i> ' : ''}${escapeHTML(message)}
            </div>
            <div class="message-info">
                <span>${getCurrentTime()}</span>
            </div>
        </div>`;
    userMessageElement.innerHTML = userMessageHTML;
    if (messagesContainer) messagesContainer.appendChild(userMessageElement);

    // Clear input and scroll
    if (messageInput) messageInput.value = '';
    autoResizeTextarea();
    scrollToBottom();

    // Show typing indicator
    if (typingIndicator) typingIndicator.style.display = 'flex';

    try {
        // Choose API endpoint based on web search mode
        const apiUrl = isWebSearchActive ? API_WEB_SEARCH_URL : API_ASK_URL;
        
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: message })
        });

        if (typingIndicator) typingIndicator.style.display = 'none';

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        const answer = data.response || "Could not retrieve an answer.";
        const sources = data.sources || [];

        appendBotMessage(answer, sources);

    } catch (error) {
        console.error('Error:', error);
        if (typingIndicator) typingIndicator.style.display = 'none';
        appendBotMessage(`Sorry, I encountered an error while processing your request. Please try again later.`);
    }

    scrollToBottom();
    if (sendMessageBtn) sendMessageBtn.disabled = false;
}

/**
 * Append a bot message to the messages container
 * @param {string} answer - The AI generated answer
 * @param {Array<Object>} sources - Array of source document metadata (e.g., [{display: "...", content_snippet: "...", metadata: {...}}])
 */
function appendBotMessage(answer, sources = []) {
    if (!messagesContainer) return; // Ensure container exists

    const botMessageElement = document.createElement('div');
    botMessageElement.className = 'message'; // No 'outgoing' class for bot messages

    let messageContentHTML = `
        <div class="message-avatar bot-avatar">
            <i class="fas fa-scale-balanced"></i>
        </div>
        <div class="message-content">
            <div class="message-text">
                ${escapeHTML(answer).replace(/\n/g, '<br>')} <!-- ✅ Escape answer and convert newlines -->
            </div>`;

    // ✅ CORRECTED: Iterate through sources array returned by the backend
    if (sources && sources.length > 0) {
        messageContentHTML += `
            <div class="sources">
                <h4>Sources Used:</h4>
                <ul>`; // Use a list for sources
        sources.forEach((source, index) => {
             // Ensure source object and required properties exist
             if (source && source.display) {
                  messageContentHTML += `<li>
                                            <strong>${escapeHTML(source.display)}</strong>
                                            ${source.content_snippet ? `<br><em>Snippet: "${escapeHTML(source.content_snippet)}"</em>` : ''}
                                          </li>`; // ✅ Escape source display and snippet
             } else {
                  console.warn("Source object missing expected structure:", source);
                  messageContentHTML += `<li>Source ${index + 1}: Details unavailable</li>`;
             }
        });
        messageContentHTML += `</ul></div>`;
    }

    messageContentHTML += `
            <div class="message-info">
                <span>${getCurrentTime()}</span>
            </div>
        </div>`;

    botMessageElement.innerHTML = messageContentHTML;
    messagesContainer.appendChild(botMessageElement);
}


// --- Event Listeners ---

// Send message on button click
if (sendMessageBtn) {
    sendMessageBtn.addEventListener('click', sendMessage);
}


// Send message on Enter key press (but not Shift+Enter)
if (messageInput) {
    messageInput.addEventListener('keypress', handleKeyPress);
    messageInput.addEventListener('input', autoResizeTextarea); // Auto-resize as user types
}


// Handle Enter key press in textarea
function handleKeyPress(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault(); // Prevent newline in textarea
        sendMessage();
    }
}

// --- File Upload Handling (STILL DEMO, NOT CONNECTED TO RAG) ---
// These elements are controlled by ui.js based on the HTML, ensure they exist
const fileUpload = document.getElementById('file-upload');
const filePreview = document.getElementById('file-preview');
const fileName = document.getElementById('file-name');
const removeFile = document.getElementById('remove-file');

// Handle file selection (UI only)
if (fileUpload) {
    fileUpload.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            uploadedFile = e.target.files[0];
            if (fileName) fileName.textContent = escapeHTML(uploadedFile.name); // ✅ Escape filename
            if (filePreview) filePreview.style.display = 'flex';
             console.log("File selected for upload (DEMO). Not sent to RAG backend.");
        }
    });
}

// Handle file removal (UI only)
if (removeFile) {
    removeFile.addEventListener('click', () => {
        uploadedFile = null;
        if (fileUpload) fileUpload.value = ''; // Clear the file input
        if (filePreview) filePreview.style.display = 'none';
         console.log("File cleared (DEMO).");
    });
}


// --- Web Search Toggle (STILL DEMO) ---
if (webSearchBtn) {
     webSearchBtn.addEventListener('click', toggleWebSearch);
}

function toggleWebSearch() {
    isWebSearchActive = !isWebSearchActive;

    if (isWebSearchActive) {
        console.log("Web Search mode toggled ON (DEMO)");
        if (webSearchBtn) webSearchBtn.style.backgroundColor = '#a68657'; // Highlight button (adjust color as per your CSS)
        if (messageInput) messageInput.placeholder = "Ask Mitra (Web Search Active - DEMO)...";
         // NOTE: Actual web search logic needs to be added to the backend and/or frontend.
         // The current sendMessage function DOES NOT PERFORM WEB SEARCH.
         // You would need to modify sendMessage to potentially call a different API or pass a flag.
    } else {
         console.log("Web Search mode toggled OFF (DEMO)");
        if (webSearchBtn) webSearchBtn.style.backgroundColor = ''; // Revert highlight
        if (messageInput) messageInput.placeholder = "Type your legal question here...";
    }

    if (messageInput) messageInput.focus();
}


// --- Utility Functions ---

/**
 * Auto-resize textarea based on content
 */
function autoResizeTextarea() {
    if (messageInput) {
        messageInput.style.height = 'auto'; // Reset height
        messageInput.style.height = (messageInput.scrollHeight) + 'px'; // Set to scroll height
    }
}

/**
 * Scroll messages container to bottom
 */
function scrollToBottom() {
    if (messagesContainer) {
        // Use a slight delay to ensure elements are rendered before scrolling
        setTimeout(() => {
             messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }, 50); // Adjust delay if needed
    }
}

/**
 * Get current time in HH:MM format
 * @returns {string} Current time
 */
function getCurrentTime() {
    const now = new Date();
    const hours = now.getHours().toString().padStart(2, '0');
    const minutes = now.getMinutes().toString().padStart(2, '0');
    return `${hours}:${minutes}`;
}

/**
 * Helper function to shorten text (used for demo history titles)
 * @param {string} text - Text to shorten
 * @param {number} maxLength - Maximum length
 * @returns {string} Shortened text
 */
function shortenText(text, maxLength) {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength).trim() + '...';
}

/**
 * Basic HTML escaping for safety when using innerHTML
 */
function escapeHTML(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
}


// --- Initialization ---
// Call this function when the chat UI is ready to be displayed
// (Your auth.js or app.js should likely call this after successful login/app load)
function initializeChatUI() {
    console.log("Initializing chat UI...");
    populateChatHistory(); // Populate sidebar history (demo)
    // Initially show the welcome screen until a new chat is started
    if (welcomeScreen) welcomeScreen.style.display = 'flex';
    if (messagesContainer) messagesContainer.style.display = 'none';
    if (chatInputContainer) chatInputContainer.style.display = 'none';

    // Add listeners for starting a new chat if not already added
    // This handles clicks on both the sidebar 'New Conversation' button
    // and the button on the welcome screen.
     if (newChatBtn && !newChatBtn.__listenerAdded) {
        newChatBtn.addEventListener('click', startNewChat);
        newChatBtn.__listenerAdded = true; // Prevent adding multiple listeners
     }
     if (startNewChatBtn && !startNewChatBtn.__listenerAdded) {
        startNewChatBtn.addEventListener('click', startNewChat);
        startNewChatBtn.__listenerAdded = true; // Prevent adding multiple listeners
     }
    // Also potentially trigger a new chat immediately if the user is already logged in
    // or this page is directly accessed after login.
    // startNewChat(); // Uncomment this line if you want to automatically start a new chat view on load
}

// Export or call initializeChatUI from your main app.js after login/load
// Example: In app.js after successful login: initializeChatUI();
// For now, let's add a temporary call to initialize it when the script loads
// In a real app, this should be triggered by your login/routing logic (e.g., after auth succeeds).
console.log("chat.js loaded. TEMPORARILY calling initializeChatUI. Replace with app logic.");
initializeChatUI();
// --- END TEMPORARY ---