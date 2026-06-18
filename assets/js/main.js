(function () {
  'use strict';

  // ===== Theme Toggle =====
  function getStoredTheme() {
    return localStorage.getItem('theme');
  }

  function getSystemTheme() {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    var icon = document.getElementById('theme-icon');
    if (icon) {
      icon.textContent = theme === 'dark' ? '\u2600' : '\u263E';
    }
  }

  function toggleTheme() {
    var current = document.documentElement.getAttribute('data-theme') || getSystemTheme();
    var next = current === 'dark' ? 'light' : 'dark';
    localStorage.setItem('theme', next);
    applyTheme(next);
  }

  // Apply theme immediately
  applyTheme(getStoredTheme() || getSystemTheme());

  document.addEventListener('DOMContentLoaded', function () {
    var btn = document.getElementById('theme-toggle');
    if (btn) {
      btn.addEventListener('click', toggleTheme);
    }

    // System theme change listener
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function () {
      if (!getStoredTheme()) {
        applyTheme(getSystemTheme());
      }
    });

    // ===== Nav Scroll Effect =====
    var nav = document.querySelector('.site-nav');
    if (nav) {
      var handleScroll = function () {
        if (window.scrollY > 60) {
          nav.classList.add('scrolled');
        } else {
          nav.classList.remove('scrolled');
        }
      };
      window.addEventListener('scroll', handleScroll, { passive: true });
      handleScroll();
    }

    var prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // ===== Scroll Reveal (Intersection Observer) =====
    var revealElements = document.querySelectorAll('.reveal, .stagger');
    if (prefersReducedMotion) {
      revealElements.forEach(function (el) {
        el.classList.add('is-visible');
      });
    } else if (revealElements.length > 0 && 'IntersectionObserver' in window) {
      var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
            observer.unobserve(entry.target);
          }
        });
      }, {
        threshold: 0.12,
        rootMargin: '0px 0px -40px 0px'
      });

      revealElements.forEach(function (el) {
        observer.observe(el);
      });
    } else {
      // Fallback: show all elements if no IntersectionObserver
      revealElements.forEach(function (el) {
        el.classList.add('is-visible');
      });
    }

    // ===== Motto: gentle drift + fade on scroll =====
    var mottoContent = document.getElementById('motto-content');
    var motto = document.getElementById('motto');
    if (mottoContent && motto && !prefersReducedMotion) {
      var ticking = false;
      var updateMotto = function () {
        var mottoHeight = motto.offsetHeight || window.innerHeight;
        var progress = Math.min(1, window.scrollY / mottoHeight); // 0 -> 1 across the hero
        mottoContent.style.transform = 'translateY(' + (progress * 24) + 'px)';
        mottoContent.style.opacity = String(Math.max(0.15, 1 - progress * 0.9));
        ticking = false;
      };
      window.addEventListener('scroll', function () {
        if (!ticking) {
          window.requestAnimationFrame(updateMotto);
          ticking = true;
        }
      }, { passive: true });
      updateMotto();
    }
  });
})();
