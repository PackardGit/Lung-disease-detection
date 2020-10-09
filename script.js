class ImageUpload extends React.Component {
  constructor(props) {
    super(props);
    this.state = { file: '', imagePreviewUrl: '', message: '', imagePreviewResponseUrl:''};
  }

  _handleSubmit(e) {
    e.preventDefault();
    return new Promise((resolve, reject) => {
      let imageFormData = new FormData();
  
      imageFormData.append('image', this.state.file);
      
      var xhr = new XMLHttpRequest();
      var z = this;
      
      xhr.open('post', '/api/image', true);
      xhr.onload = function () {
        if (this.status == 200) {
          var msg = JSON.parse(this.response)
          console.log("**" + msg.pred.name + msg.pred.prob)
          var prob = Math.round(msg.pred.prob)
          z.setState({message:`Recognized "${msg.pred.name}", probability: ${prob}%.`,
					  imagePreviewResponseUrl: 'image.jpg'})
		  
          resolve(this.response);
        } else {
          reject(this.statusText);
        }
      };
      
      xhr.send(imageFormData);
  
    });
  }

  _handleImageChange(e) {
    e.preventDefault();
    let reader = new FileReader();
    let file = e.target.files[0];

    reader.onloadend = () => {
      this.setState({
        file: file,
        message: '',
        imagePreviewUrl: reader.result,
        imagePreviewResponseUrl:''});	
    };
	
    reader.readAsDataURL(file);
	
  }

  render() {
    let { imagePreviewUrl } = this.state;
    let $imagePreview = null;
    if (imagePreviewUrl) {
      $imagePreview = React.createElement("img", { src: imagePreviewUrl });
    } else {
      $imagePreview = React.createElement("div", { className: "previewText" }, "Please select an Image for Preview");
    }
	
    let { imagePreviewResponseUrl } = this.state;
    let $imgPreviewResult = null;
    if (imagePreviewResponseUrl) {
      $imgPreviewResult = React.createElement("img", { src: imagePreviewResponseUrl });
    } else {
      $imgPreviewResult = React.createElement("div", { className: "previewTextResponse" }, "Result Image");
    }

    return (
      React.createElement("div", { className: "previewComponent" },
      React.createElement("form", { onSubmit: e => this._handleSubmit(e) },
      React.createElement("input", { className: "fileInput",
        type: "file",
        onChange: e => this._handleImageChange(e) }),
      React.createElement("button", { className: "submitButton",
        type: "submit",
        onClick: e => this._handleSubmit(e) }, "Predict a Disease")),
      React.createElement("div", { className: "imgPreview" },$imagePreview),
	  React.createElement("div", { className: "imgPreviewResult" },$imgPreviewResult),
	  React.createElement("div", { className: "TextResponse" }, this.state.message),
      React.createElement("span", {}, this.state.message),
	  

      )
      );
  }}


ReactDOM.render(React.createElement(ImageUpload, null), document.getElementById("mainApp"));