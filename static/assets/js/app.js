document.addEventListener('DOMContentLoaded', function() {
    // Populate chat history
    populateChatHistory();
    
    // Add event listeners
    setupEventListeners();
    
    // Check for dark mode preference
    checkDarkModePreference();
});

/**
 * Set up all event listeners
 */
function setupEventListeners() {
    // Auth related
    loginForm.addEventListener('submit', handleLogin);
    
    // UI related
    menuToggle.addEventListener('click', toggleSidebar);
    darkModeToggle.addEventListener('click', toggleDarkMode);
    
    // Chat related
    sendMessageBtn.addEventListener('click', sendMessage);
    messageInput.addEventListener('keydown', handleKeyPress);
    startNewChatBtn.addEventListener('click', startNewChat);
    newChatBtn.addEventListener('click', startNewChat);
    webSearchBtn.addEventListener('click', toggleWebSearch);
    
    // Auto-resize textarea
    messageInput.addEventListener('input', autoResizeTextarea);
    
    // Modal related
    const learnMoreBtn = document.querySelector('.hero-btns .btn-secondary');
    if (learnMoreBtn) {
        learnMoreBtn.addEventListener('click', (e) => {
            e.preventDefault();
            handleLearnMoreClick();
        });
    }
    
    if (modalClose) {
        modalClose.addEventListener('click', () => closeModal(learnMoreModal));
    }
    
    if (modalCloseBtn) {
        modalCloseBtn.addEventListener('click', () => closeModal(learnMoreModal));
    }
    
    // Close modal when clicking outside
    window.addEventListener('click', (e) => {
        if (e.target === learnMoreModal) {
            closeModal(learnMoreModal);
        }
    });
    
    // Escape key closes modals
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && learnMoreModal.classList.contains('active')) {
            closeModal(learnMoreModal);
        }
    });
    
    // Additional interactions
    
    // Handle window resize
    window.addEventListener('resize', handleResize);
    
    // Handle profile click
    document.querySelector('.user-profile').addEventListener('click', handleProfileClick);
}

/**
 * Handle window resize
 */
function handleResize() {
    // Close sidebar on mobile when window is resized
    if (window.innerWidth <= 768 && sidebar.classList.contains('active')) {
        sidebar.classList.remove('active');
    }
}

/**
 * Handle profile click
 */
function handleProfileClick() {
    // Create a profile dropdown menu
    const profileMenu = document.createElement('div');
    profileMenu.className = 'profile-menu';
    profileMenu.innerHTML = `
        <div class="profile-menu-item">
            <i class="fas fa-user"></i>
            <span>Profile</span>
        </div>
        <div class="profile-menu-item">
            <i class="fas fa-cog"></i>
            <span>Settings</span>
        </div>
        <div class="profile-menu-item" id="logout-btn">
            <i class="fas fa-sign-out-alt"></i>
            <span>Logout</span>
        </div>
    `;
    
    // Position the menu
    const userProfile = document.querySelector('.user-profile');
    const rect = userProfile.getBoundingClientRect();
    
    profileMenu.style.position = 'absolute';
    profileMenu.style.bottom = `${window.innerHeight - rect.top + 10}px`;
    profileMenu.style.left = `${rect.left}px`;
    profileMenu.style.width = `${rect.width}px`;
    profileMenu.style.backgroundColor = 'var(--primary-color)';
    profileMenu.style.border = '1px solid rgba(255, 255, 255, 0.1)';
    profileMenu.style.borderRadius = 'var(--border-radius)';
    profileMenu.style.boxShadow = 'var(--shadow)';
    profileMenu.style.zIndex = '100';
    profileMenu.style.overflow = 'hidden';
    
    // Style menu items
    const menuItems = profileMenu.querySelectorAll('.profile-menu-item');
    menuItems.forEach(item => {
        item.style.padding = '0.8rem 1rem';
        item.style.display = 'flex';
        item.style.alignItems = 'center';
        item.style.gap = '0.8rem';
        item.style.cursor = 'pointer';
        item.style.transition = 'var(--transition)';
        item.style.color = 'var(--light-text)';
        
        item.addEventListener('mouseenter', () => {
            item.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
        });
        
        item.addEventListener('mouseleave', () => {
            item.style.backgroundColor = 'transparent';
        });
    });
    
    // Add logout functionality
    profileMenu.querySelector('#logout-btn').addEventListener('click', () => {
        logoutUser();
        profileMenu.remove();
    });
    
    // Add the menu to the body
    document.body.appendChild(profileMenu);
    
    // Remove the menu when clicking outside
    function handleClickOutside(e) {
        if (!profileMenu.contains(e.target) && !userProfile.contains(e.target)) {
            profileMenu.remove();
            document.removeEventListener('click', handleClickOutside);
        }
    }
    
    // Set a timeout before adding the event listener to prevent immediate removal
    setTimeout(() => {
        document.addEventListener('click', handleClickOutside);
    }, 10);
}