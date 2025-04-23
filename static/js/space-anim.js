/**
 * Space Animation for Cosmic Finance
 * Creates a static cosmic background with colored regions that respond to mouse movement
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing space animation');
    const canvas = document.getElementById('galaxyCanvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Set canvas size to window size
    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    // Mouse tracking for dynamic effects
    let mouseX = canvas.width / 2;
    let mouseY = canvas.height / 2;
    
    window.addEventListener('mousemove', function(e) {
        mouseX = e.clientX;
        mouseY = e.clientY;
    });
    
    // Define color palettes for different regions
    const colorPalettes = [
        ['#00f8e7', '#00d4ff', '#00bfff', '#00aaff', '#0095ff'],  // Cosmic Blue
        ['#76ff03', '#b0ff57', '#ccff90', '#f4ff81', '#f9fbe7'],  // Toxic Green
        ['#ff1744', '#ff5252', '#ff867f', '#ff4081', '#f50057']   // Radiation Red
    ];
    
    // Create star objects
    class Star {
        constructor() {
            this.palette = colorPalettes[Math.floor(Math.random() * colorPalettes.length)];
            this.reset();
        }
        
        reset() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.size = Math.random() * 1.5 + 0.5;
            this.color = this.palette[Math.floor(Math.random() * this.palette.length)];
            this.alpha = Math.random() * 0.8 + 0.2;
            this.twinkleSpeed = Math.random() * 0.03 + 0.01;
            this.twinklePhase = Math.random() * Math.PI * 2;
        }
        
        update() {
            // Update twinkle effect
            this.twinklePhase += this.twinkleSpeed;
            
            // Mouse influence - subtle movement
            const dx = (mouseX - canvas.width/2) * 0.0005;
            const dy = (mouseY - canvas.height/2) * 0.0005;
            
            this.x += dx;
            this.y += dy;
            
            // Wrap around edges
            if (this.x < 0) this.x = canvas.width;
            if (this.x > canvas.width) this.x = 0;
            if (this.y < 0) this.y = canvas.height;
            if (this.y > canvas.height) this.y = 0;
        }
        
        draw() {
            // Apply twinkle effect
            const twinkleFactor = Math.sin(this.twinklePhase) * 0.5 + 0.5;
            ctx.globalAlpha = this.alpha * twinkleFactor;
            
            ctx.fillStyle = this.color;
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            ctx.fill();
        }
    }
    
    // Create color region objects
    class ColorRegion {
        constructor() {
            this.palette = colorPalettes[Math.floor(Math.random() * colorPalettes.length)];
            this.reset();
        }
        
        reset() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.radius = Math.random() * 200 + 100;
            this.color = this.palette[Math.floor(Math.random() * this.palette.length)];
            this.alpha = Math.random() * 0.05 + 0.02; // Very subtle
            this.pulseSpeed = Math.random() * 0.005 + 0.001;
            this.pulsePhase = Math.random() * Math.PI * 2;
        }
        
        update() {
            // Pulsating effect
            this.pulsePhase += this.pulseSpeed;
            
            // Mouse influence - move towards mouse
            const dx = (mouseX - this.x) * 0.0002;
            const dy = (mouseY - this.y) * 0.0002;
            
            this.x += dx;
            this.y += dy;
            
            // Keep within bounds
            if (this.x < -this.radius) this.x = canvas.width + this.radius;
            if (this.x > canvas.width + this.radius) this.x = -this.radius;
            if (this.y < -this.radius) this.y = canvas.height + this.radius;
            if (this.y > canvas.height + this.radius) this.y = -this.radius;
        }
        
        draw() {
            // Apply pulse effect
            const pulseFactor = Math.sin(this.pulsePhase) * 0.2 + 0.8;
            const currentRadius = this.radius * pulseFactor;
            
            const gradient = ctx.createRadialGradient(
                this.x, this.y, 0,
                this.x, this.y, currentRadius
            );
            
            gradient.addColorStop(0, this.color);
            gradient.addColorStop(1, 'rgba(0,0,0,0)');
            
            ctx.globalAlpha = this.alpha;
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(this.x, this.y, currentRadius, 0, Math.PI * 2);
            ctx.fill();
        }
    }
    
    // Create objects
    const stars = Array(300).fill().map(() => new Star());
    const colorRegions = Array(15).fill().map(() => new ColorRegion());
    
    // Mouse position tracker for smooth movement
    let targetMouseX = mouseX;
    let targetMouseY = mouseY;
    
    // Animation function
    function animate() {
        // Create trail effect with semi-transparent black
        ctx.fillStyle = 'rgba(5, 5, 15, 0.1)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Smooth mouse movement
        mouseX += (targetMouseX - mouseX) * 0.1;
        mouseY += (targetMouseY - mouseY) * 0.1;
        
        // Update and draw color regions (background)
        colorRegions.forEach(region => {
            region.update();
            region.draw();
        });
        
        // Update and draw stars (foreground)
        stars.forEach(star => {
            star.update();
            star.draw();
        });
        
        // Reset global alpha
        ctx.globalAlpha = 1;
        
        // Continue animation
        requestAnimationFrame(animate);
    }
    
    // Track actual mouse position
    window.addEventListener('mousemove', function(e) {
        targetMouseX = e.clientX;
        targetMouseY = e.clientY;
    });
    
    // Start animation
    animate();
});
