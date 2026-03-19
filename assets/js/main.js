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

    // ===== Fade-in on Scroll (Intersection Observer) =====
    var fadeElements = document.querySelectorAll('.fade-in');
    if (fadeElements.length > 0 && 'IntersectionObserver' in window) {
      var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible');
            observer.unobserve(entry.target);
          }
        });
      }, {
        threshold: 0.1,
        rootMargin: '0px 0px -40px 0px'
      });

      fadeElements.forEach(function (el) {
        observer.observe(el);
      });
    } else {
      // Fallback: show all elements if no IntersectionObserver
      fadeElements.forEach(function (el) {
        el.classList.add('visible');
      });
    }
  });
})();
