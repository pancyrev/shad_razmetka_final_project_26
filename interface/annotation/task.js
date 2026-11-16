exports.Task = extend(TolokaHandlebarsTask, function(options) {
    TolokaHandlebarsTask.call(this, options);
}, {
    onRender: function() {
        this.getField('annotations').getEditor().zoom = 50;
    }
});
function extend(ParentClass, constructorFunction, prototypeHash) {
    constructorFunction = constructorFunction || function() {};
    prototypeHash = prototypeHash || {};
    if (ParentClass) {
        constructorFunction.prototype = Object.create(ParentClass.prototype);
    }
    for (var key in prototypeHash) {
        constructorFunction.prototype[key] = prototypeHash[key];
    }
    return constructorFunction;
}