// Mobile menu toggle
const navToggle = document.querySelector('.nav-toggle');
const mobileMenu = document.querySelector('.mobile-menu');

if (navToggle && mobileMenu) {
    navToggle.addEventListener('click', () => {
        mobileMenu.classList.toggle('active');
        navToggle.classList.toggle('active');
    });
}

// Close mobile menu on link click
if (mobileMenu) {
    mobileMenu.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', () => {
            mobileMenu.classList.remove('active');
            if (navToggle) navToggle.classList.remove('active');
        });
    });
}

// Scroll-triggered fade-in animations
const observerOptions = {
    threshold: 0.15,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

// Apply fade-in to sections
document.querySelectorAll('.pillar-content, .hero-title, .hero-subtitle, .credentials-text, .cta-title, .cta-text, .cta-links').forEach(el => {
    el.classList.add('fade-in');
    observer.observe(el);
});

// Stagger animation delays for pillar sections
document.querySelectorAll('.pillar').forEach((pillar, index) => {
    const content = pillar.querySelector('.pillar-content');
    if (content) {
        content.style.transitionDelay = `${index * 0.05}s`;
    }
});

// Header background on scroll
const header = document.querySelector('.header');
let lastScroll = 0;

window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;

    if (currentScroll > 100) {
        header.style.boxShadow = '0 1px 8px rgba(0,0,0,0.04)';
    } else {
        header.style.boxShadow = 'none';
    }

    lastScroll = currentScroll;
}, { passive: true });

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    if (anchor.id === 'open-roles-btn') return;
    anchor.addEventListener('click', (e) => {
        e.preventDefault();
        const target = document.querySelector(anchor.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});

// Email tooltip click toggle
const openRolesBtn = document.getElementById('open-roles-btn');
const emailTooltip = document.getElementById('email-tooltip');

if (openRolesBtn && emailTooltip) {
    openRolesBtn.addEventListener('click', (e) => {
        e.preventDefault();
        emailTooltip.classList.toggle('active');
    });

    document.addEventListener('click', (e) => {
        if (!e.target.closest('.tooltip-wrapper')) {
            emailTooltip.classList.remove('active');
        }
    });

    document.querySelectorAll('.tooltip-email').forEach(el => {
        el.addEventListener('click', () => {
            const email = el.dataset.email;
            navigator.clipboard.writeText(email).then(() => {
                el.classList.add('copied');
                const icon = el.querySelector('.copy-icon');
                const originalHTML = icon.outerHTML;
                icon.outerHTML = '<svg class="copy-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
                setTimeout(() => {
                    const newIcon = el.querySelector('.copy-icon');
                    newIcon.outerHTML = originalHTML;
                    el.classList.remove('copied');
                }, 1500);
            });
        });
    });
}

// Checkerboard hover highlight effect
const checkerCanvas = document.createElement('canvas');
checkerCanvas.className = 'checker-highlight';
checkerCanvas.style.cssText = 'position:fixed;inset:0;pointer-events:none;z-index:0;';
document.body.prepend(checkerCanvas);

const ctx = checkerCanvas.getContext('2d');
let mouseX = -100, mouseY = -100;
let animId = null;

function resizeCanvas() {
    checkerCanvas.width = window.innerWidth;
    checkerCanvas.height = window.innerHeight;
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

function drawHighlight() {
    ctx.clearRect(0, 0, checkerCanvas.width, checkerCanvas.height);
    const gridSize = 20;
    const radius = 80;

    const startCol = Math.max(0, Math.floor((mouseX - radius) / gridSize));
    const endCol = Math.min(Math.ceil(checkerCanvas.width / gridSize), Math.ceil((mouseX + radius) / gridSize));
    const startRow = Math.max(0, Math.floor((mouseY - radius) / gridSize));
    const endRow = Math.min(Math.ceil(checkerCanvas.height / gridSize), Math.ceil((mouseY + radius) / gridSize));

    for (let row = startRow; row < endRow; row++) {
        for (let col = startCol; col < endCol; col++) {
            if ((row + col) % 2 === 0) continue;
            const cx = col * gridSize + gridSize / 2;
            const cy = row * gridSize + gridSize / 2;
            const dist = Math.sqrt((cx - mouseX) ** 2 + (cy - mouseY) ** 2);
            if (dist < radius) {
                const intensity = 1 - (dist / radius);
                const scale = 1 + intensity * 0.3;
                const alpha = intensity * 0.25;
                const size = gridSize * scale;
                const offset = (size - gridSize) / 2;
                ctx.fillStyle = `rgba(200, 200, 200, ${alpha})`;
                ctx.fillRect(col * gridSize - offset, row * gridSize - offset, size, size);
            }
        }
    }
    animId = requestAnimationFrame(drawHighlight);
}

document.addEventListener('mousemove', (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
    if (!animId) animId = requestAnimationFrame(drawHighlight);
});

document.addEventListener('mouseleave', () => {
    mouseX = -100;
    mouseY = -100;
});

drawHighlight();

// Home logo multi-line burst effect
const homeLogo = document.querySelector('.home-logo-text');
if (homeLogo) {
    let burstInterval = null;

    function createBurst() {
        const lineCount = 6;
        for (let i = 0; i < lineCount; i++) {
            const left = document.createElement('span');
            const right = document.createElement('span');
            left.className = 'burst-line burst-line--left';
            right.className = 'burst-line burst-line--right';

            const thickness = Math.random() * 3 + 1;
            const offsetY = (Math.random() - 0.5) * 30;
            const delay = Math.random() * 0.05;
            const duration = 0.5 + Math.random() * 0.4;

            const style = `
                height: ${thickness}px;
                top: calc(50% + ${offsetY}px);
                animation-delay: ${delay}s;
                animation-duration: ${duration}s;
            `;
            left.style.cssText = style;
            right.style.cssText = style;

            homeLogo.appendChild(left);
            homeLogo.appendChild(right);

            // Remove after animation ends
            setTimeout(() => {
                left.remove();
                right.remove();
            }, (delay + duration) * 1000 + 100);
        }
    }

    homeLogo.addEventListener('mouseenter', () => {
        createBurst();
        burstInterval = setInterval(createBurst, 300);
    });

    homeLogo.addEventListener('mouseleave', () => {
        clearInterval(burstInterval);
        burstInterval = null;
    });
}
