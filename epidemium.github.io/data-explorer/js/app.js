DE = window.DE || {};

DE.App = function(options) {
    options || (options = {});
    _.extend(this, options);
};

_.extend(DE.App.prototype, Backbone.Events, {
  initialize: function() {
    this.uploadView = new DE.UploadView({
      app: this
    });

    this.on('rowsAdded', this.rowsAdded);
  },
  rowsAdded: function(rows) {
    this.rows = rows;

    this.uploadView.remove();

    this.tableView = new DE.TableView({
      data: rows
    });
  }
});