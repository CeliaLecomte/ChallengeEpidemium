DE = window.DE || {};

DE.UploadView = Backbone.View.extend({
  el: '.js-upload-view',
  initialize: function(opts) {
    this.app = opts.app;
    this.render();
  },
  render: function() {
    var that = this;

    this.dropzone = new Dropzone(document.body, {
      url: '/',
      autoProcessQueue: false,
      // stop the body from being clickable
      clickable: false,
      addedfile: function(file) {
        var data = Papa.parse(file, {
          complete: function(results) {
            var rows = new DE.RowCollection(results.data);
            that.app.trigger('rowsAdded', rows);
          }
        });
      }
    });
  }
});