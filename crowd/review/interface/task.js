exports.Task = extend(TolokaHandlebarsTask, function(options) {
  TolokaHandlebarsTask.call(this, options);
}, {
  onRender: function() {
    var root = this.getDOMElement();
    root = root && (root[0] || root);
    var figure = root.querySelector('.bbox-review-image');
    var image = figure.querySelector('img');
    var svg = figure.querySelector('svg');
    var error = root.querySelector('.bbox-review-error');
    var answers = root.querySelector('.bbox-review-answers');
    var status = figure.getAttribute('data-worker-status');
    var labels = {
      FOUND: 'Бабочки есть, все обведены рамками',
      NONE: 'Бабочек нет',
      LOAD_ERROR: 'Ошибка загрузки изображения',
      UNCERTAIN: 'Сомневаюсь в объекте'
    };

    function invalid(message) {
      error.textContent = message;
      answers.style.display = 'none';
    }

    if (!Object.prototype.hasOwnProperty.call(labels, status)) {
      invalid('Некорректный ответ разметчика. Сообщите заказчику.');
      return;
    }
    root.querySelector('.bbox-review-worker-status').textContent =
      'Ответ разметчика: ' + labels[status] + '.';

    var boxes;
    try {
      boxes = JSON.parse(figure.getAttribute('data-boxes-json'));
    } catch (e) {
      invalid('Рамки не прочитались. Сообщите заказчику.');
      return;
    }
    if (!Array.isArray(boxes)) {
      invalid('Неверный формат рамок. Сообщите заказчику.');
      return;
    }
    for (var i = 0; i < boxes.length; i++) {
      var b = boxes[i];
      if (!b || ![b.left, b.top, b.width, b.height].every(function(n) {
        return typeof n === 'number' && isFinite(n);
      }) || b.left < 0 || b.top < 0 || b.width <= 0 || b.height <= 0 ||
          b.left + b.width > 1 || b.top + b.height > 1) {
        invalid('Некорректные координаты рамки. Сообщите заказчику.');
        return;
      }
      var rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.setAttribute('x', b.left);
      rect.setAttribute('y', b.top);
      rect.setAttribute('width', b.width);
      rect.setAttribute('height', b.height);
      svg.appendChild(rect);
    }

    function syncSize() {
      var width = image.getBoundingClientRect().width;
      var height = image.getBoundingClientRect().height;
      if (width > 0 && height > 0) {
        figure.style.width = width + 'px';
        svg.style.width = width + 'px';
        svg.style.height = height + 'px';
      }
    }
    image.addEventListener('load', syncSize);
    if (image.complete && image.naturalWidth) syncSize();
    image.onerror = function() {
      error.textContent = 'Фото не открылось. Повторите загрузку; если ошибка сохранится, выберите «Изображение не загрузилось».';
    };
    if (image.complete && !image.naturalWidth) image.onerror();
    root.querySelector('.bbox-review-zoom').addEventListener('click', function() {
      var large = figure.classList.toggle('bbox-review-image-large');
      this.textContent = large ? 'Уменьшить фото' : 'Увеличить фото';
      syncSize();
    });
  },
  onDestroy: function() {}
});

function extend(ParentClass, constructorFunction, prototypeHash) {
  constructorFunction.prototype = Object.create(ParentClass.prototype);
  for (var key in prototypeHash) {
    if (Object.prototype.hasOwnProperty.call(prototypeHash, key)) {
      constructorFunction.prototype[key] = prototypeHash[key];
    }
  }
  return constructorFunction;
}
