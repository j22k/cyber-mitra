const header = document.querySelector('.about-header');
const mobileMenuToggle = document.getElementById('mobile-menu-toggle');
const nav = document.querySelector('.nav');

// Handle scroll event for header styling
document.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
        header.classList.add('scrolled');
    } else {
        header.classList.remove('scrolled');
    }
});

// Handle mobile menu toggle
mobileMenuToggle.addEventListener('click', () => {
    nav.classList.toggle('active');
    
    // Change the icon based on menu state
    if (nav.classList.contains('active')) {
        mobileMenuToggle.innerHTML = '<i class="fas fa-times"></i>';
    } else {
        mobileMenuToggle.innerHTML = '<i class="fas fa-bars"></i>';
    }
});

// Close mobile menu when clicking outside
document.addEventListener('click', (e) => {
    if (nav.classList.contains('active') && 
        !nav.contains(e.target) && 
        !mobileMenuToggle.contains(e.target)) {
        nav.classList.remove('active');
        mobileMenuToggle.innerHTML = '<i class="fas fa-bars"></i>';
    }
});

// Handle scroll animations
document.addEventListener('DOMContentLoaded', () => {
    // Animate elements when they come into view
    const animateElements = document.querySelectorAll('.section-title, .divider, .section-text, .feature, .achievement-card, .value, .mission-card, .vision-card');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '0';
                entry.target.style.transform = 'translateY(20px)';
                
                // Animate the element
                setTimeout(() => {
                    entry.target.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }, 100);
                
                // Stop observing the element after animation
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.2 });
    
    // Start observing elements
    animateElements.forEach(element => {
        // Set initial state
        element.style.opacity = '0';
        element.style.transform = 'translateY(20px)';
        
        observer.observe(element);
    });
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            // Close mobile menu if open
            if (nav.classList.contains('active')) {
                nav.classList.remove('active');
                mobileMenuToggle.innerHTML = '<i class="fas fa-bars"></i>';
            }
            
            const targetId = this.getAttribute('href');
            
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                // Calculate header height for offset
                const headerHeight = header.offsetHeight;
                
                window.scrollTo({
                    top: targetElement.offsetTop - headerHeight - 20,
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Add active class to nav link based on current scroll position
    const sections = document.querySelectorAll('section');
    const navLinks = document.querySelectorAll('.nav-link');
    
    function setActiveNavLink() {
        const headerHeight = header.offsetHeight;
        let current = '';
        
        sections.forEach((section) => {
            const sectionTop = section.offsetTop - headerHeight - 100;
            const sectionHeight = section.clientHeight;
            
            if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
                current = section.getAttribute('id');
            }
        });
        
        navLinks.forEach((link) => {
            link.classList.remove('active');
            
            if (link.getAttribute('href') === `#${current}` || 
                (current === '' && link.getAttribute('href') === '#hero')) {
                link.classList.add('active');
            }
        });
    }
    
    // Initial call to set active nav link
    setActiveNavLink();
    
    // Update active nav link on scroll
    window.addEventListener('scroll', setActiveNavLink);
    
    // Counter animation for achievements
    function animateCounters() {
        const counters = document.querySelectorAll('.counter');
        
        counters.forEach(counter => {
            const target = parseInt(counter.getAttribute('data-target'));
            const duration = 2000; // ms
            const step = Math.ceil(target / (duration / 16)); // 16ms is approx 1 frame at 60fps
            
            let count = 0;
            
            const updateCount = () => {
                count += step;
                
                if (count < target) {
                    counter.textContent = count;
                    requestAnimationFrame(updateCount);
                } else {
                    counter.textContent = target;
                }
            };
            
            updateCount();
        });
    }
    
    // Initial animations
    setTimeout(() => {
        document.querySelector('.hero-content').style.opacity = '1';
        document.querySelector('.hero-content').style.transform = 'translateX(0)';
        
        setTimeout(() => {
            document.querySelector('.hero-image').style.opacity = '1';
            document.querySelector('.hero-image').style.transform = 'translateX(0)';
        }, 300);
    }, 300);
});

// Preload images for better performance
function preloadImages() {
    const images = [
        './assets/images/ai-illustration.svg',
        './assets/images/legal-background.jpg'
    ];
    
    images.forEach(src => {
        const img = new Image();
        img.src = src;
    });
}

// Call preloadImages on page load
window.addEventListener('load', preloadImages);